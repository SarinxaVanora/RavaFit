import json,struct
from pathlib import Path
import numpy as np

from multi_primitive_glb import ProductionGLB, scatter_domain_attribute


def _align4(b: bytearray):
    while len(b)%4:b.append(0)


def _make_glb(path: Path, *, separate_domains=False):
    blob=bytearray(); views=[]; accessors=[]
    def add(arr,component_type,atype):
        arr=np.asarray(arr);_align4(blob);off=len(blob);raw=arr.tobytes();blob.extend(raw)
        vi=len(views);views.append({'buffer':0,'byteOffset':off,'byteLength':len(raw)})
        ai=len(accessors);accessors.append({'bufferView':vi,'componentType':component_type,'count':len(arr),'type':atype})
        if np.issubdtype(arr.dtype,np.floating) and arr.ndim==2:
            accessors[-1]['min']=arr.min(0).astype(float).tolist();accessors[-1]['max']=arr.max(0).astype(float).tolist()
        return ai
    p0=np.asarray([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],np.float32)
    p1=np.asarray([[2,0,0],[3,0,0],[3,1,0],[2,1,0]],np.float32)
    n=np.tile(np.asarray([[0,0,1]],np.float32),(4,1));uv=np.asarray([[0,0],[1,0],[1,1],[0,1]],np.float32)
    j=np.zeros((4,4),np.uint16);w=np.zeros((4,4),np.float32);w[:,0]=1
    pos0=add(p0,5126,'VEC3');norm0=add(n,5126,'VEC3');uv0=add(uv,5126,'VEC2');j0=add(j,5123,'VEC4');w0=add(w,5126,'VEC4')
    if separate_domains:
        pos1=add(p1,5126,'VEC3');norm1=add(n,5126,'VEC3');uv1=add(uv,5126,'VEC2');j1=add(j,5123,'VEC4');w1=add(w,5126,'VEC4')
    else:
        pos1,norm1,uv1,j1,w1=pos0,norm0,uv0,j0,w0
    i0=add(np.asarray([0,1,2],np.uint16),5123,'SCALAR');i1=add(np.asarray([0,2,3],np.uint16),5123,'SCALAR')
    attrs0={'POSITION':pos0,'NORMAL':norm0,'TEXCOORD_0':uv0,'JOINTS_0':j0,'WEIGHTS_0':w0}
    attrs1={'POSITION':pos1,'NORMAL':norm1,'TEXCOORD_0':uv1,'JOINTS_0':j1,'WEIGHTS_0':w1}
    js={'asset':{'version':'2.0'},'buffers':[{'byteLength':len(blob)}],'bufferViews':views,'accessors':accessors,
        'materials':[{'name':'/mt_test.mtrl'}],
        'meshes':[{'name':'mesh 1','primitives':[{'attributes':attrs0,'indices':i0,'material':0},{'attributes':attrs1,'indices':i1,'material':0}]}],
        'nodes':[{'name':'j_kosi'},{'name':'garment','mesh':0,'skin':0}],
        'skins':[{'joints':[0]}], 'scenes':[{'nodes':[1]}], 'scene':0}
    jb=json.dumps(js,separators=(',',':')).encode();jb+=b' '*((-len(jb))%4);bb=bytes(blob);bb+=b'\0'*((-len(bb))%4)
    out=bytearray(struct.pack('<4sII',b'glTF',2,12+8+len(jb)+8+len(bb)));out+=struct.pack('<II',len(jb),0x4E4F534A)+jb;out+=struct.pack('<II',len(bb),0x004E4942)+bb
    path.write_bytes(out)


def test_shared_vertex_domain_uses_all_primitives(tmp_path):
    path=tmp_path/'shared.glb';_make_glb(path,separate_domains=False)
    d=ProductionGLB(path).data('mesh 1')
    assert d['V'].shape==(4,3)
    assert d['F'].shape==(2,3)
    assert {tuple(row) for row in d['F'].tolist()}=={(0,1,2),(0,2,3)}
    assert d['_primitive_count']==2
    assert len(d['_vertex_domains'])==1


def test_separate_vertex_domains_are_concatenated_and_scattered(tmp_path):
    source=tmp_path/'separate.glb';out=tmp_path/'out.glb';_make_glb(source,separate_domains=True)
    view=ProductionGLB(source);d=view.data('mesh 1')
    assert d['V'].shape==(8,3)
    assert d['F'].shape==(2,3)
    assert tuple(d['F'][1])==(4,6,7)
    new=d['V'].copy();new[:,2]+=0.125
    report=scatter_domain_attribute(view,'mesh 1',new,'POSITION')
    view.save(out)
    final=ProductionGLB(out).data('mesh 1')
    np.testing.assert_allclose(final['V'],new,atol=1e-7)
    assert report['primitive_count']==2
    assert report['vertex_domain_count']==2
    assert len(report['domains'])==2
