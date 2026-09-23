from __future__ import annotations
import io, json, struct, zipfile, zlib, hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import numpy as np

# FFXIV MDL vertex declarations used by body models.
USAGE = {0:'POSITION',1:'BONE_WEIGHT',2:'BONE_INDEX',3:'NORMAL',4:'TEXCOORD',5:'FLOW',6:'BINORMAL',7:'COLOR'}
TYPE = {0:'FLOAT1',1:'FLOAT2',2:'FLOAT3',3:'FLOAT4',5:'UBYTE4',6:'SHORT2',7:'SHORT4',8:'UBYTE4N',9:'SHORT2N',10:'SHORT4N',13:'HALF2',14:'HALF4',17:'UBYTE8'}

MODEL_DATA_FMT='<8hHBBHBBffHhBBBB6h'
MODEL_DATA_KEYS=['MeshCount','AttributeCount','MeshPartCount','MaterialCount','BoneCount','BoneSetCount','ShapeCount','ShapePartCount','ShapeDataCount','LoDCount','Flags1','ElementIdCount','TerrainShadowMeshCount','Flags2','ModelClip','ShadowClip','FurnitureBBoxCount','TerrainShadowPartCount','Flags3','BgChange','BgCrest','NeckMorph','BoneSetSize','Unknown13','Patch72','Unknown15','Unknown16','Unknown17']

SLOT_MAP={'top':'Chest','dwn':'Legs','glv':'Hands','sho':'Feet'}


def read_blocks(data: bytes, pos: int, count: int) -> bytes:
    outs=[]
    for _ in range(count):
        start=pos
        while pos<len(data) and data[pos]==0: pos+=1
        if pos>=len(data) or data[pos]!=16:
            raise ValueError(('bad sqpack block header',pos,data[pos:pos+16].hex()))
        h0,h1,comp,decomp=struct.unpack_from('<IIII',data,pos)
        if h0!=16 or h1!=0: raise ValueError(('bad sqpack block',h0,h1))
        pos+=16
        n=decomp if comp==32000 else comp
        chunk=data[pos:pos+n]; pos+=n
        if comp==32000: out=chunk
        else:
            try: out=zlib.decompress(chunk)
            except zlib.error: out=zlib.decompress(chunk,-15)
        if len(out)!=decomp:
            try: out=zlib.decompress(chunk,-15)
            except Exception: pass
        if len(out)!=decomp: raise ValueError(('block size',len(out),decomp))
        pos=start+((pos-start+127)//128)*128
        outs.append(out)
    return b''.join(outs)


def dec_type3(data: bytes) -> bytes:
    if len(data)<24: raise ValueError('short type3')
    headerLength,fileType,decompressedSize,buf1,buf2,version=struct.unpack_from('<6i',data,0)
    if fileType!=3: raise ValueError(('not model type3',fileType))
    p=24
    vertexInfoSize,modelDataSize=struct.unpack_from('<2I',data,p);p+=8
    def a3(fmt='<3I'):
        nonlocal p
        vals=struct.unpack_from(fmt,data,p);p+=struct.calcsize(fmt);return vals
    vertexBufferSizes=a3(); edgeSizes=a3(); indexSizes=a3()
    vertexInfoComp,modelDataComp=struct.unpack_from('<2I',data,p);p+=8
    vbComp=a3();edgeComp=a3();idxComp=a3()
    vertexInfoOffset,modelDataOffset=struct.unpack_from('<2I',data,p);p+=8
    vbOff=a3();edgeOff=a3();idxOff=a3()
    viBI,mdBI=struct.unpack_from('<2H',data,p);p+=4
    vbBI=a3('<3H');edgeBI=a3('<3H');idxBI=a3('<3H')
    viBC,mdBC=struct.unpack_from('<2H',data,p);p+=4
    vbBC=a3('<3H');edgeBC=a3('<3H');idxBC=a3('<3H')
    meshCount,materialCount=struct.unpack_from('<2H',data,p);p+=4
    lodCount,flags=struct.unpack_from('<2B',data,p);p+=2
    padding=data[p:p+2]
    end=headerLength
    vi=read_blocks(data,end+vertexInfoOffset,viBC); md=read_blocks(data,end+modelDataOffset,mdBC)
    vbs=[read_blocks(data,end+vbOff[i],vbBC[i]) for i in range(3)]
    edges=[read_blocks(data,end+edgeOff[i],edgeBC[i]) for i in range(3)]
    idxs=[read_blocks(data,end+idxOff[i],idxBC[i]) for i in range(3)]
    cur=68+len(vi)+len(md); vb_uoffs=[];ib_uoffs=[];vb_real=[];ib_real=[];body=bytearray(vi+md)
    for i in range(3):
        vb_uoffs.append(cur);body+=vbs[i];cur+=len(vbs[i]);vb_real.append(len(vbs[i]));body+=edges[i];cur+=len(edges[i]);ib_uoffs.append(cur);body+=idxs[i];cur+=len(idxs[i]);ib_real.append(len(idxs[i]))
    h=bytearray();h+=struct.pack('<i',version);h+=struct.pack('<i',len(vi));h+=struct.pack('<i',len(md));h+=struct.pack('<HH',meshCount,materialCount);h+=struct.pack('<3I',*vb_uoffs);h+=struct.pack('<3I',*ib_uoffs);h+=struct.pack('<3I',*vb_real);h+=struct.pack('<3I',*ib_real);h+=struct.pack('<BB',lodCount,flags)+padding
    if len(h)!=68: raise ValueError('header')
    return bytes(h)+bytes(body)


def stored_member_offset(pkg_path: str|Path, z: zipfile.ZipFile, info: zipfile.ZipInfo):
    if info.compress_type!=zipfile.ZIP_STORED:return None
    with open(pkg_path,'rb') as f:
        f.seek(info.header_offset);h=f.read(30)
    if h[:4]!=b'PK\x03\x04':return None
    nlen,xlen=struct.unpack_from('<HH',h,26)
    return info.header_offset+30+nlen+xlen


def _read_cstrings(block: bytes) -> list[str]:
    return [x.decode('utf-8','replace') for x in block.split(b'\0') if x]

def _path_string(block: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(block): return ''
    end=block.find(b'\0',offset)
    if end<0:end=len(block)
    return block[offset:end].decode('utf-8','replace')


def _decode_attribute(data: bytes, base: int, stride: int, offset: int, typ: int, n: int):
    specs={
        0:('<f4',1),1:('<f4',2),2:('<f4',3),3:('<f4',4),
        5:('u1',4),6:('<i2',2),7:('<i2',4),8:('u1',4),
        9:('<i2',2),10:('<i2',4),13:('<f2',2),14:('<f2',4),17:('u1',8),
    }
    if typ not in specs: raise ValueError(('unsupported vertex type',typ))
    dtype,nc=specs[typ]; dt=np.dtype(dtype)
    arr=np.ndarray((n,nc),dtype=dt,buffer=data,offset=base+offset,strides=(stride,dt.itemsize)).copy()
    if typ==8: return arr.astype(np.float32)/255.0
    if typ in (9,10): return np.maximum(arr.astype(np.float32)/32767.0,-1.0)
    if typ in (13,14): return arr.astype(np.float32)
    return arr


def _parse_model_data(data: bytes, off: int):
    radius=struct.unpack_from('<f',data,off)[0]; off+=4
    vals=struct.unpack_from(MODEL_DATA_FMT,data,off); off+=struct.calcsize(MODEL_DATA_FMT)
    return {'Radius':radius,**dict(zip(MODEL_DATA_KEYS,vals))},off



def _index_buffer_candidate_is_valid(data: bytes, base: int, infos: list[dict[str,Any]]) -> bool:
    if base < 0:
        return False
    for info in infos:
        count=max(0,int(info.get('icount',0)))
        vertex_count=max(0,int(info.get('vcount',0)))
        index_offset=max(0,int(info.get('idxoff',0)))
        if count <= 0:
            continue
        byte_start=base+index_offset*2
        byte_end=byte_start+count*2
        if byte_start < 0 or byte_end > len(data):
            return False
        values=np.frombuffer(data,dtype='<u2',count=count,offset=byte_start)
        if len(values) and (vertex_count <= 0 or int(values.max()) >= vertex_count):
            return False
    return True


def _resolve_lod_index_buffer_base(data: bytes, lod: dict[str,Any], infos: list[dict[str,Any]], lods: list[dict[str,Any]]) -> tuple[int,str|None]:
    """Resolve the physical start of an MDL index buffer without assuming exporter padding is truthful.

    Most XIV MDLs put the index payload exactly at the header's IndexDataOffset.  Some authored/imported
    MDLs, however, keep VertexDataOffset+VertexDataSize in that field while inserting or omitting a few
    bytes at the vertex/index boundary.  Penumbra/the game can still load those files, but treating the
    header value as an unconditional numpy offset makes perfectly valid submeshes look like 65k indices.

    Keep the declared offset as first authority.  Only if its complete mesh ranges are impossible do we
    recover the buffer from the enclosing section end and declared IndexDataSize, then use a tiny bounded
    search as a final compatibility fallback.  Every candidate must prove that all mesh index values are
    inside that mesh's vertex range before it is accepted.
    """
    declared=int(lod.get('ioff',0))
    size=max(0,int(lod.get('isize',0)))
    if size <= 0 or not infos:
        return declared,None
    if _index_buffer_candidate_is_valid(data,declared,infos):
        return declared,None

    # Find the next real data block. Zero-sized LOD offsets are end markers in some imported MDLs and
    # must not be treated as physical boundaries. If nothing follows this index buffer, EOF is authority.
    later=[]
    for other in lods:
        vsize=max(0,int(other.get('vsize',0))); isize=max(0,int(other.get('isize',0)))
        voff=int(other.get('voff',0)); ioff=int(other.get('ioff',0))
        if vsize > 0 and voff > declared:
            later.append(voff)
        if isize > 0 and ioff > declared:
            later.append(ioff)
    section_end=min(later) if later else len(data)
    anchored=section_end-size
    if abs(anchored-declared) <= 64 and _index_buffer_candidate_is_valid(data,anchored,infos):
        delta=anchored-declared
        return anchored,f'index buffer boundary corrected by {delta:+d} byte(s) from MDL header offset'

    # Rare malformed/imported MDLs can disagree by a handful of bytes without an exact section-size
    # anchor. Fail closed outside a deliberately tiny window rather than guessing into unrelated data.
    for distance in range(1,65):
        for candidate in (declared-distance,declared+distance):
            if _index_buffer_candidate_is_valid(data,candidate,infos):
                delta=candidate-declared
                return candidate,f'index buffer boundary recovered by {delta:+d} byte(s) after validating all mesh indices'
    return declared,None


def _vertex_buffer_candidate_is_sane(data: bytes, base: int, infos: list[dict[str,Any]], decls: list[list[tuple]]) -> bool:
    """Conservatively reject byte-shifted vertex streams before applying a recovered LOD delta."""
    if base < 0:
        return False
    checked=0
    for mi,info in enumerate(infos):
        if mi >= len(decls):
            break
        n=max(0,int(info.get('vcount',0)))
        if n <= 0:
            continue
        position_decl=next((d for d in decls[mi] if int(d[3]) == 0),None)
        if position_decl is None:
            continue
        block,doff,typ,_,_=position_decl
        block=int(block)
        strides=info.get('stride') or []
        offsets=info.get('vdo') or []
        if block < 0 or block >= len(strides) or block >= len(offsets) or int(strides[block]) <= 0:
            return False
        sample=min(n,128)
        try:
            values=np.asarray(_decode_attribute(data,base+int(offsets[block]),int(strides[block]),int(doff),int(typ),sample),dtype=np.float32)[:,:3]
        except Exception:
            return False
        if not len(values) or not np.isfinite(values).all():
            return False
        # Player/equipment model coordinates are small. A byte-shifted FLOAT3 stream usually becomes
        # infinities/NaNs or enormous exponents; use a deliberately generous ceiling to avoid rejecting
        # legitimate oversized props while still refusing obviously misaligned data.
        if float(np.max(np.abs(values))) > 1000.0:
            return False
        checked+=1
    return checked > 0

def parse_rigged_mdl(data: bytes) -> dict[str,Any]:
    """Parse LOD0 standard meshes into a B14 reference view with global joint indices."""
    if len(data)<100: raise ValueError('short mdl')
    version16=struct.unpack_from('<H',data,0)[0]
    mdl_version=6 if version16>=6 else 5
    meshCount=struct.unpack_from('<H',data,12)[0]
    if meshCount<=0 or 68+136*meshCount>=len(data): raise ValueError(('meshcount',meshCount))
    decls=[]
    for j in range(meshCount):
        q=68+j*136; ds=[]; lim=68+(j+1)*136
        while q<lim:
            block=data[q]
            if block==255: break
            doff,typ,usage,count=data[q+1:q+5]
            ds.append((block,doff,typ,usage,count)); q+=8
        decls.append(ds)

    off=68+136*meshCount
    pathCount,pathSize=struct.unpack_from('<ii',data,off); off+=8
    pathBlock=data[off:off+pathSize]; off+=pathSize
    md,off=_parse_model_data(data,off)
    if md['Flags2']!=0:
        raise ValueError(('unsupported Flags2',md['Flags2']))
    off += md['ElementIdCount']*32

    lods=[]
    for _ in range(3):
        std_i,std_c=struct.unpack_from('<HH',data,off);off+=4
        mdlr,texr=struct.unpack_from('<ff',data,off);off+=8
        water=struct.unpack_from('<HH',data,off);off+=4
        shadow=struct.unpack_from('<HH',data,off);off+=4
        terrain=struct.unpack_from('<HH',data,off);off+=4
        fog=struct.unpack_from('<HH',data,off);off+=4
        edge_size,edge_off,u6,u7,vsize,isize,voff,ioff=struct.unpack_from('<8i',data,off);off+=32
        lods.append(dict(std_i=std_i,std_c=std_c,water=water,shadow=shadow,terrain=terrain,fog=fog,vsize=vsize,isize=isize,voff=voff,ioff=ioff))

    infos=[]
    for l in lods:
        count=l['std_c']+l['water'][1]+l['shadow'][1]+l['fog'][1]
        arr=[]
        for _ in range(count):
            vals=struct.unpack_from('<iihhhhiiiiBBBB',data,off);off+=36
            arr.append(dict(vcount=vals[0],icount=vals[1],mat=vals[2],part_index=vals[3],part_count=vals[4],bone_set=vals[5],idxoff=vals[6],vdo=[vals[7],vals[8],vals[9]],stride=[vals[10],vals[11],vals[12]],stream_count=vals[13]))
        infos.append(arr)

    index_buffer_bases=[];vertex_buffer_bases=[];index_buffer_notes=[]
    for lod_index,(lod,arr) in enumerate(zip(lods,infos)):
        resolved,note=_resolve_lod_index_buffer_base(data,lod,arr,lods)
        resolved=int(resolved);delta=resolved-int(lod['ioff'])
        index_buffer_bases.append(resolved)
        declared_vertex=int(lod['voff']);resolved_vertex=declared_vertex
        if lod_index == 0 and delta:
            shifted_vertex=declared_vertex+delta
            declared_sane=_vertex_buffer_candidate_is_sane(data,declared_vertex,arr,decls)
            shifted_sane=_vertex_buffer_candidate_is_sane(data,shifted_vertex,arr,decls)
            # Prefer the declared vertex offset whenever it is sane. Only carry the recovered index
            # delta into the vertex streams when the header position is demonstrably unusable and the
            # shifted position decodes sane player/equipment geometry. This also covers exporters that
            # shift only the index payload rather than the complete LOD data block.
            if not declared_sane and shifted_sane:
                resolved_vertex=shifted_vertex
                index_buffer_notes.append(f'LOD {lod_index}: vertex buffer boundary corrected by {delta:+d} byte(s) from MDL header offset')
        vertex_buffer_bases.append(resolved_vertex)
        if note:index_buffer_notes.append(f'LOD {lod_index}: {note}')

    # The MDL metadata after mesh descriptors is authoritative for submeshes and shape keys.
    # Otopop (and other Penumbra bodies) use both: attributes select authored submeshes, while
    # shapes replace indices with alternate vertices already present in the model vertex buffers.
    attribute_offsets=list(struct.unpack_from('<'+'I'*md['AttributeCount'],data,off)) if md['AttributeCount'] else [];off+=md['AttributeCount']*4
    off += lods[0]['terrain'][1]*20
    submeshes=[]
    for part_index in range(md['MeshPartCount']):
        struct_offset=off
        index_offset,index_count,attribute_mask,bone_start,bone_count=struct.unpack_from('<IIIHH',data,off);off+=16
        submeshes.append({'part_index':part_index,'struct_offset':struct_offset,'index_offset':int(index_offset),'index_count':int(index_count),'attribute_mask':int(attribute_mask),'bone_start_index':int(bone_start),'bone_count':int(bone_count)})
    for arr2 in infos:
        for info2 in arr2:
            pi=max(0,int(info2['part_index']));pc=max(0,int(info2['part_count']))
            info2['submeshes']=submeshes[pi:pi+pc]
    off += md['TerrainShadowPartCount']*12
    material_offsets=list(struct.unpack_from('<'+'I'*md['MaterialCount'],data,off)) if md['MaterialCount'] else [];off+=md['MaterialCount']*4
    bone_offsets=list(struct.unpack_from('<'+'I'*md['BoneCount'],data,off)) if md['BoneCount'] else [];off+=md['BoneCount']*4

    bone_sets=[]
    bone_set_start=off
    bone_set_end=bone_set_start + md['BoneSetSize']*2 + md['BoneSetCount']*4
    if mdl_version>=6:
        meta=[]
        for _ in range(md['BoneSetCount']):
            a,b=struct.unpack_from('<hh',data,off);off+=4; meta.append((a,b))
        for _,bc in meta:
            vals=list(struct.unpack_from('<'+'h'*bc,data,off)) if bc else []; off+=2*bc
            if bc%2: off+=2
            bone_sets.append(vals)
        off=bone_set_end
    else:
        for _ in range(md['BoneSetCount']):
            vals=list(struct.unpack_from('<64h',data,off));off+=128
            bc=struct.unpack_from('<i',data,off)[0];off+=4
            bone_sets.append(vals[:bc])
        off=bone_set_end

    shape_rows=[]
    for _ in range(md['ShapeCount']):
        string_offset,*ranges=struct.unpack_from('<I3H3H',data,off);off+=16
        shape_rows.append({'string_offset':int(string_offset),'mesh_start':[int(x) for x in ranges[:3]],'mesh_count':[int(x) for x in ranges[3:]]})
    shape_mesh_rows=[]
    for _ in range(md['ShapePartCount']):
        mesh_index_offset,shape_value_count,shape_value_offset=struct.unpack_from('<III',data,off);off+=12
        shape_mesh_rows.append({'mesh_index_offset':int(mesh_index_offset),'shape_value_count':int(shape_value_count),'shape_value_offset':int(shape_value_offset)})
    shape_values=[]
    for _ in range(md['ShapeDataCount']):
        base_indices_index,replacing_vertex_index=struct.unpack_from('<HH',data,off);off+=4
        shape_values.append((int(base_indices_index),int(replacing_vertex_index)))

    attribute_names=[_path_string(pathBlock,x) for x in attribute_offsets]
    joint_names=[_path_string(pathBlock,x) for x in bone_offsets]
    material_names=[_path_string(pathBlock,x) for x in material_offsets]
    shape_names=[_path_string(pathBlock,row['string_offset']) for row in shape_rows]
    if len(joint_names)!=md['BoneCount'] or any(not name for name in joint_names):
        # Older catalogues were parsed successfully from c-string order; retain that fallback for odd models.
        strings=_read_cstrings(pathBlock);a0=md['AttributeCount'];b0=a0;b1=b0+md['BoneCount'];m1=b1+md['MaterialCount']
        joint_names=strings[b0:b1];material_names=strings[b1:m1]
    if len(joint_names)!=md['BoneCount']:
        raise ValueError(('bone path count',len(joint_names),md['BoneCount']))

    # Keep a lightweight map of the standard mesh/index ranges for every authored LOD.  Solver
    # geometry still uses LOD0 only; this metadata exists so surgical native-MDL operations such
    # as visibility toggles can modify the same semantic part at distance without rebuilding MDL.
    lod_standard_mesh_records=[]
    for lod_index,(lod,arr) in enumerate(zip(lods,infos)):
        start=int(lod['std_i']); count=int(lod['std_c'])
        if start+count>len(arr): start=0
        records=[]
        for standard_ordinal,mi in enumerate(range(start,start+count)):
            info=arr[mi]
            lod_submeshes=[]
            for local_part_ordinal,sm in enumerate(info.get('submeshes',[])):
                attrs=[attribute_names[bit] for bit in range(min(32,len(attribute_names))) if (int(sm['attribute_mask']) & (1<<bit)) and attribute_names[bit]]
                lod_submeshes.append({
                    'submesh_ordinal':int(local_part_ordinal),
                    'part_index':int(sm['part_index']),
                    'struct_offset':int(sm['struct_offset']),
                    'local_index_offset':int(sm['index_offset'])-int(info['idxoff']),
                    'index_count':int(sm['index_count']),
                    'attribute_mask':int(sm['attribute_mask']),
                    'attributes':attrs,
                })
            attrs=sorted({a for sm in lod_submeshes for a in sm.get('attributes',[]) if a})
            records.append({
                'standard_ordinal':int(standard_ordinal),
                'mesh_index':int(mi),
                'source_index_offset':int(info['idxoff']),
                'index_count':int(info['icount']),
                'material_index':int(info['mat']),
                'material':material_names[info['mat']] if 0<=info['mat']<len(material_names) else '',
                'attributes':attrs,
                'submeshes':lod_submeshes,
            })
        lod_standard_mesh_records.append({
            'lod':int(lod_index),
            'index_buffer_offset':int(index_buffer_bases[lod_index]),
            'records':records,
        })

    shapes={}
    for name,row in zip(shape_names,shape_rows):
        if not name:continue
        parts=[]
        st=row['mesh_start'][0];ct=row['mesh_count'][0]
        for sm in shape_mesh_rows[st:st+ct]:
            sv=shape_values[sm['shape_value_offset']:sm['shape_value_offset']+sm['shape_value_count']]
            parts.append({'mesh_index_offset':sm['mesh_index_offset'],'values':sv})
        shapes[name]={'lod0_parts':parts}

    l=lods[0]; arr=infos[0]; start=l['std_i']; count=l['std_c']
    if start+count>len(arr): start=0
    positions=[];normals=[];uv0s=[];joints=[];weights=[];indices=[];mesh_records=[];materials=[]
    vbase=0; ibase=0
    warnings=list(index_buffer_notes)
    max_influences=8
    for mi in range(start,start+count):
        info=arr[mi]; ds=decls[mi] if mi<len(decls) else []
        decl_by_usage={}
        for d in ds:
            decl_by_usage.setdefault(d[3],[]).append(d)
        if 0 not in decl_by_usage: raise ValueError(('no position decl',mi))
        n=info['vcount']
        P=np.empty((n,3),np.float32); N=np.zeros((n,3),np.float32); UV=np.zeros((n,2),np.float32)
        J=np.full((n,max_influences),65535,np.uint16); W=np.zeros((n,max_influences),np.float32)
        hasN=3 in decl_by_usage; hasUV=4 in decl_by_usage; hasBI=2 in decl_by_usage; hasBW=1 in decl_by_usage
        posd=decl_by_usage[0][0]; nd=decl_by_usage.get(3,[None])[0]; uvd=decl_by_usage.get(4,[None])[0]; bid=decl_by_usage.get(2,[None])[0]; bwd=decl_by_usage.get(1,[None])[0]
        def attr(d):
            block,doff,typ,usage,cnt=d
            if block>2 or info['stride'][block]<=0: raise ValueError(('bad stream',block,info['stride']))
            return _decode_attribute(data,vertex_buffer_bases[0]+info['vdo'][block],info['stride'][block],doff,typ,n)
        P[:]=attr(posd)[:,:3]
        if nd is not None:
            N[:]=attr(nd).astype(np.float32)[:,:3]
        if uvd is not None:
            UV[:]=attr(uvd).astype(np.float32)[:,:2]
        if bid is not None and bwd is not None:
            lj=np.asarray(attr(bid),dtype=np.int64)
            ww=np.asarray(attr(bwd),dtype=np.float32)
            k=min(max_influences,lj.shape[1],ww.shape[1]); lj=lj[:,:k]; ww=ww[:,:k]
            bs=np.asarray(bone_sets[info['bone_set']] if 0<=info['bone_set']<len(bone_sets) else [],dtype=np.int64)
            valid_local=(lj>=0)&(lj<len(bs))&(ww>0)
            mapped=np.full(lj.shape,65535,dtype=np.int64)
            if len(bs): mapped[valid_local]=bs[lj[valid_local]]
            valid_global=valid_local&(mapped>=0)&(mapped<len(joint_names))
            J[:,:k]=np.where(valid_global,mapped,65535).astype(np.uint16)
            W[:,:k]=np.where(valid_global,ww,0.0)
            sw=W.sum(1); good=sw>1e-12; W[good]/=sw[good,None]
            bad=int(np.count_nonzero((ww>0)&~valid_global))
            if bad: warnings.append(f'mesh {mi}: {bad} weighted bone references could not be mapped')
        ii=np.frombuffer(data,dtype='<u2',count=info['icount'],offset=index_buffer_bases[0]+info['idxoff']*2).astype(np.uint32)
        if len(ii) and int(ii.max())>=n: raise ValueError(('bad index',mi,int(ii.max()),n))
        mesh_submeshes=[]
        for sm in info.get('submeshes',[]):
            local_start=int(sm['index_offset'])-int(info['idxoff'])
            attrs=[attribute_names[bit] for bit in range(min(32,len(attribute_names))) if (int(sm['attribute_mask']) & (1<<bit)) and attribute_names[bit]]
            mesh_submeshes.append({'part_index':int(sm['part_index']),'struct_offset':int(sm['struct_offset']),'local_index_offset':local_start,'index_count':int(sm['index_count']),'attribute_mask':int(sm['attribute_mask']),'attributes':attrs})
        positions.append(P); normals.append(N); uv0s.append(UV); joints.append(J); weights.append(W); indices.append(ii+vbase)
        mesh_records.append({'mesh_index':mi,'source_index_offset':int(info['idxoff']),'vertex_offset':vbase,'vertex_count':n,'index_offset':ibase,'index_count':len(ii),'material_index':int(info['mat']),'bone_set_index':int(info['bone_set']),'has_normal':hasN,'has_uv0':hasUV,'has_skin':hasBI and hasBW,'submeshes':mesh_submeshes})
        materials.append(material_names[info['mat']] if 0<=info['mat']<len(material_names) else '')
        vbase+=n; ibase+=len(ii)
    if not positions: raise ValueError('no standard geometry')
    P=np.concatenate(positions); N=np.concatenate(normals); UV=np.concatenate(uv0s); J=np.concatenate(joints); W=np.concatenate(weights); I=np.concatenate(indices)
    # Normalise normals when present.
    ln=np.linalg.norm(N,axis=1); good=ln>1e-12; N[good]/=ln[good,None]
    solver_compatible=bool(len(P) and len(I) and np.any(np.linalg.norm(N,axis=1)>0) and np.any(np.linalg.norm(UV,axis=1)>0) and np.all(W.sum(axis=1)>0.99))
    return {
        'positions':P,'normals':N,'uv0':UV,'joints':J,'weights':W,'indices':I,
        'joint_names':joint_names,'materials':materials,'mesh_records':mesh_records,'mdl_version':mdl_version,
        'attribute_names':attribute_names,'shapes':shapes,'submeshes':submeshes,'lod0_index_buffer_offset':int(index_buffer_bases[0]),
        'lod_standard_mesh_records':lod_standard_mesh_records,
        'model_data':md,'warnings':warnings[:1000],'warning_count':len(warnings),'solver_compatible':solver_compatible,
        'bounds_min':P.min(0).astype(float).tolist(),'bounds_max':P.max(0).astype(float).tolist(),
    }




def bake_model_view_mdl(data: bytes, model_view: dict[str,Any]|None) -> tuple[bytes,dict[str,Any]]:
    """Bake a catalogue model view into MDL index data while preserving mesh/vertex/index counts.

    Shape keys in FFXIV MDLs are index substitutions into alternate authored vertices.  Attribute
    toggles select submeshes.  Baking those choices here gives Penumbra's exporter and the final
    native-body graft the same concrete target topology without reconstructing the model.
    """
    view=model_view or {}
    attribute_states={str(k):bool(v) for k,v in (view.get('attributes') or {}).items()}
    requested_shapes=[str(x) for x in (view.get('shapes') or []) if str(x)]
    aliases={str(k):str(v) for k,v in (view.get('shape_aliases') or {}).items()}
    strict=bool(view.get('strict',False))
    if not attribute_states and not requested_shapes:return data,{'applied':False,'shapes':[],'disabled_submeshes':0}
    ref=parse_rigged_mdl(data);out=bytearray(data);ioff=int(ref['lod0_index_buffer_offset'])
    found=[]
    for requested in requested_shapes:
        actual=requested if requested in ref.get('shapes',{}) else aliases.get(requested,requested)
        shape=(ref.get('shapes') or {}).get(actual)
        if shape is None:
            if strict:raise ValueError(f'RBODY model view requires missing shape {requested!r}')
            continue
        found.append(requested)
        for part in shape.get('lod0_parts',[]):
            mesh_offset=int(part['mesh_index_offset'])
            for base_index,replacement in part.get('values',[]):
                absolute=ioff+(mesh_offset+int(base_index))*2
                if absolute<0 or absolute+2>len(out):raise ValueError(f'RBODY shape {requested!r} index points outside MDL')
                struct.pack_into('<H',out,absolute,int(replacement))
    disabled=0;degenerate_triangles=0
    for mr in ref.get('mesh_records',[]):
        source_index_offset=int(mr.get('source_index_offset',0))
        for sm in mr.get('submeshes',[]):
            names=set(str(x) for x in sm.get('attributes',[]))
            if not any(name in names and not state for name,state in attribute_states.items()):continue
            disabled+=1;local=max(0,int(sm.get('local_index_offset',0)));count=max(0,int(sm.get('index_count',0)))
            # Preserve the exact index count for native grafting; hidden triangles become degenerate.
            for rel in range(0,count-(count%3),3):
                absolute=ioff+(source_index_offset+local+rel)*2
                if absolute<0 or absolute+6>len(out):raise ValueError('RBODY attribute submesh index range points outside MDL')
                first=struct.unpack_from('<H',out,absolute)[0]
                struct.pack_into('<HHH',out,absolute,first,first,first);degenerate_triangles+=1
    return bytes(out),{'applied':True,'shapes':found,'disabled_submeshes':disabled,'degenerate_triangles':degenerate_triangles,'attributes':attribute_states}


def apply_model_view(ref: dict[str,Any], model_view: dict[str,Any]|None) -> dict[str,Any]:
    """Bake a catalogue shape/attribute view into solver geometry without mutating the raw parsed MDL."""
    view=model_view or {}
    attribute_states={str(k):bool(v) for k,v in (view.get('attributes') or {}).items()}
    enabled_shapes=[str(x) for x in (view.get('shapes') or []) if str(x)]
    aliases={str(k):str(v) for k,v in (view.get('shape_aliases') or {}).items()}
    strict=bool(view.get('strict',False))
    if not attribute_states and not enabled_shapes:return ref

    available_shapes=ref.get('shapes') or {}
    resolved_shapes=[]
    missing=[]
    for requested in enabled_shapes:
        actual=requested if requested in available_shapes else aliases.get(requested,requested)
        if actual not in available_shapes: missing.append(requested);continue
        resolved_shapes.append((requested,actual,available_shapes[actual]))
    if strict and missing:raise ValueError(f"RBODY model view requires missing shape(s): {', '.join(missing)}")

    out_pos=[];out_nor=[];out_uv=[];out_j=[];out_w=[];out_i=[];out_records=[];out_mats=[]
    vbase=0;ibase=0
    for ordinal,(mr,mat) in enumerate(zip(ref['mesh_records'],ref['materials'])):
        a=int(mr['vertex_offset']);b=a+int(mr['vertex_count']);c=int(mr['index_offset']);d=c+int(mr['index_count'])
        local=np.asarray(ref['indices'][c:d],dtype=np.int64)-a
        source_index_offset=int(mr.get('source_index_offset',0))
        for _,_,shape in resolved_shapes:
            for part in shape.get('lod0_parts',[]):
                if int(part.get('mesh_index_offset',-1))!=source_index_offset:continue
                for base_index,replacement in part.get('values',[]):
                    if 0<=int(base_index)<len(local):local[int(base_index)]=int(replacement)

        submeshes=list(mr.get('submeshes') or [])
        if attribute_states and submeshes:
            kept=[]
            for sm in submeshes:
                names=set(str(x) for x in sm.get('attributes',[]))
                # A Penumbra attribute explicitly disabled by this catalogue state hides the submesh.
                if any(name in names and not state for name,state in attribute_states.items()):continue
                lo=max(0,int(sm.get('local_index_offset',0)));hi=min(len(local),lo+max(0,int(sm.get('index_count',0))))
                if hi>lo:kept.append(local[lo:hi])
            local=np.concatenate(kept) if kept else np.zeros((0,),dtype=np.int64)

        if len(local)%3:local=local[:len(local)-(len(local)%3)]
        if not len(local):
            # Keep an empty mesh record so material ordinal mapping remains truthful.
            out_records.append({'mesh_index':int(mr.get('mesh_index',ordinal)),'source_index_offset':source_index_offset,'vertex_offset':vbase,'vertex_count':0,'index_offset':ibase,'index_count':0,'material_index':int(mr.get('material_index',-1)),'bone_set_index':int(mr.get('bone_set_index',-1)),'has_normal':bool(mr.get('has_normal')),'has_uv0':bool(mr.get('has_uv0')),'has_skin':bool(mr.get('has_skin')),'submeshes':[]})
            out_mats.append(mat);continue
        valid=(local>=0)&(local<int(mr['vertex_count']))
        if not np.all(valid):raise ValueError(f"RBODY model view produced an invalid replacement vertex for mesh {mr.get('mesh_index',ordinal)}")
        used=np.unique(local)
        remap=np.full(int(mr['vertex_count']),-1,dtype=np.int64);remap[used]=np.arange(len(used),dtype=np.int64)
        compact=remap[local].astype(np.uint32)
        out_pos.append(ref['positions'][a:b][used]);out_nor.append(ref['normals'][a:b][used]);out_uv.append(ref['uv0'][a:b][used]);out_j.append(ref['joints'][a:b][used]);out_w.append(ref['weights'][a:b][used]);out_i.append(compact+vbase)
        out_records.append({'mesh_index':int(mr.get('mesh_index',ordinal)),'source_index_offset':source_index_offset,'vertex_offset':vbase,'vertex_count':len(used),'index_offset':ibase,'index_count':len(compact),'material_index':int(mr.get('material_index',-1)),'bone_set_index':int(mr.get('bone_set_index',-1)),'has_normal':bool(mr.get('has_normal')),'has_uv0':bool(mr.get('has_uv0')),'has_skin':bool(mr.get('has_skin')),'submeshes':[]})
        out_mats.append(mat);vbase+=len(used);ibase+=len(compact)

    if not out_pos:raise ValueError('RBODY model view removed all standard geometry')
    P=np.concatenate(out_pos);N=np.concatenate(out_nor);UV=np.concatenate(out_uv);J=np.concatenate(out_j);W=np.concatenate(out_w);I=np.concatenate(out_i) if out_i else np.zeros((0,),np.uint32)
    result=dict(ref);result.update({'positions':P,'normals':N,'uv0':UV,'joints':J,'weights':W,'indices':I,'mesh_records':out_records,'materials':out_mats,'applied_model_view':{'attributes':attribute_states,'shapes':[x[0] for x in resolved_shapes]}})
    result['bounds_min']=P.min(0).astype(float).tolist();result['bounds_max']=P.max(0).astype(float).tolist()
    result['solver_compatible']=bool(len(P) and len(I) and np.any(np.linalg.norm(N,axis=1)>0) and np.any(np.linalg.norm(UV,axis=1)>0) and np.all(W.sum(axis=1)>0.99))
    return result

def dense_weights(ref: dict[str,Any]) -> np.ndarray:
    J=ref['joints']; W=ref['weights']; out=np.zeros((len(J),len(ref['joint_names'])),np.float64)
    rows=np.arange(len(J))
    for k in range(J.shape[1]):
        jj=J[:,k].astype(np.int64); ww=W[:,k].astype(np.float64); ok=(jj>=0)&(jj<len(ref['joint_names']))&(ww>0)
        np.add.at(out,(rows[ok],jj[ok]),ww[ok])
    sw=out.sum(1); good=sw>1e-12; out[good]/=sw[good,None]
    return out


def solver_view(ref: dict[str,Any]) -> dict[str,Any]:
    return {'V':ref['positions'].astype(np.float64),'F':ref['indices'].reshape(-1,3).astype(np.int64),'UV':ref['uv0'].astype(np.float64),'N':ref['normals'].astype(np.float64),'W':dense_weights(ref),'joint_names':list(ref['joint_names'])}


def hash_solver_identity(ref: dict[str,Any]) -> str:
    h=hashlib.sha256()
    for a in (ref['positions'].astype('<f4',copy=False),ref['indices'].astype('<u4',copy=False),ref['normals'].astype('<f4',copy=False),ref['uv0'].astype('<f4',copy=False),ref['joints'].astype('<u2',copy=False),ref['weights'].astype('<f4',copy=False)):
        h.update(a.tobytes())
    h.update('\0'.join(ref['joint_names']).encode('utf8'))
    return h.hexdigest()

ACCESSORY_MATERIAL_TOKENS=('pierc','undies','bra','crop','betterpits','fishnet','outfit')
SMALLCLOTHES_SUPPORT_EXCLUDE_TOKENS=('pierc','jewel','ring','necklace','earring','bracelet')

def body_surface_view(ref: dict[str,Any], rig_joint_names: list[str]|None=None, surface_mode: str="body") -> dict[str,Any]:
    """Return the target surface used for body correspondence."""
    mode=str(surface_mode or "body").casefold()
    if mode not in {"body","smallclothes"}:raise ValueError(f"Unknown RBODY support surface mode {surface_mode!r}")
    chosen=[]
    if mode=="smallclothes":
        for mr,mat in zip(ref['mesh_records'],ref['materials']):
            ml=mat.lower().replace('\\','/')
            if mr['vertex_count']<=0 or mr['index_count']<=0:continue
            if any(token in ml for token in SMALLCLOTHES_SUPPORT_EXCLUDE_TOKENS):continue
            chosen.append((mr,mat))
    else:
        for mr,mat in zip(ref['mesh_records'],ref['materials']):
            ml=mat.lower().replace('\\','/')
            if mr['vertex_count']<=0 or mr['index_count']<=0:continue
            if 'b0001' in ml and not any(t in ml for t in ACCESSORY_MATERIAL_TOKENS):chosen.append((mr,mat))
        if not chosen:
            for mr,mat in zip(ref['mesh_records'],ref['materials']):
                if mr['vertex_count']>0 and mr['index_count']>0:chosen=[(mr,mat)];break
    if not chosen:
        raise ValueError(f"RBODY {mode} support surface contains no eligible geometry after model-view/material filtering")
    pos=[];nor=[];uv=[];js=[];ws=[];idx=[];records=[];vbase=0
    for mr,mat in chosen:
        a=mr['vertex_offset'];b=a+mr['vertex_count'];c=mr['index_offset'];d=c+mr['index_count']
        pos.append(ref['positions'][a:b]);nor.append(ref['normals'][a:b]);uv.append(ref['uv0'][a:b]);js.append(ref['joints'][a:b]);ws.append(ref['weights'][a:b]);idx.append(ref['indices'][c:d]-a+vbase)
        records.append({'mesh_index':int(mr.get('mesh_index',len(records))),'material':mat,'vertex_offset':vbase,'vertex_count':mr['vertex_count'],'index_count':mr['index_count'],'face_offset':int(sum(record['index_count'] for record in records)//3),'face_count':int(mr['index_count']//3)})
        vbase+=mr['vertex_count']
    P=np.concatenate(pos) if pos else np.zeros((0,3),np.float32);N=np.concatenate(nor) if nor else np.zeros((0,3),np.float32);UV=np.concatenate(uv) if uv else np.zeros((0,2),np.float32);J=np.concatenate(js) if js else np.zeros((0,8),np.uint16);Wc=np.concatenate(ws) if ws else np.zeros((0,8),np.float32);I=np.concatenate(idx) if idx else np.zeros((0,),np.uint32)
    names=list(ref['joint_names'])
    if rig_joint_names is None: rig_joint_names=names
    ridx={n:i for i,n in enumerate(rig_joint_names)}
    dense=np.zeros((len(P),len(rig_joint_names)),np.float64);rows=np.arange(len(P));
    for k in range(J.shape[1] if J.ndim==2 else 0):
        jj=J[:,k].astype(np.int64); ww=Wc[:,k].astype(np.float64);ok=(jj>=0)&(jj<len(names))&(ww>0)
        if not np.any(ok):continue
        srcrows=rows[ok];srcj=jj[ok];srcw=ww[ok]
        mapped=np.array([ridx.get(names[int(x)],-1) for x in srcj],dtype=np.int64);ok2=mapped>=0
        np.add.at(dense,(srcrows[ok2],mapped[ok2]),srcw[ok2])
    sw=dense.sum(1);good=sw>1e-12;dense[good]/=sw[good,None]
    return {'V':P.astype(np.float64),'F':I.reshape(-1,3).astype(np.int64),'UV':UV.astype(np.float64),'N':N.astype(np.float64),'W':dense,'joint_names':list(rig_joint_names),'mesh_records':records,'surface_mode':mode}
