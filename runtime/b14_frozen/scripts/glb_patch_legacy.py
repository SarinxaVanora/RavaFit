import struct, json, numpy as np
from pathlib import Path
COMP={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
NCOMP={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}
class GLBEditor:
    def __init__(self,path):
        b=Path(path).read_bytes();magic,version,length=struct.unpack_from('<4sII',b,0)
        if magic!=b'glTF': raise ValueError('Not a GLB')
        self.version=version;off=12;chunks=[]
        while off<length:
            clen,ctype=struct.unpack_from('<II',b,off);off+=8;data=b[off:off+clen];off+=clen;chunks.append((ctype,data))
        self.js=json.loads(chunks[0][1].decode('utf-8').rstrip('\x00 '))
        self.bin=bytearray(next(data for ctype,data in chunks if ctype==0x004E4942))
    def accessor(self,index,copy=True):
        a=self.js['accessors'][index];bv=self.js['bufferViews'][a['bufferView']];dt=np.dtype(COMP[a['componentType']]).newbyteorder('<');nc=NCOMP[a['type']];count=a['count']
        off=bv.get('byteOffset',0)+a.get('byteOffset',0);stride=bv.get('byteStride',dt.itemsize*nc)
        arr=np.ndarray((count,nc),dtype=dt,buffer=self.bin,offset=off,strides=(stride,dt.itemsize))
        return arr.copy() if copy else arr
    def write_accessor(self,index,values):
        arr=self.accessor(index,copy=False);v=np.asarray(values,dtype=arr.dtype)
        if arr.shape!=v.shape: raise ValueError(f'accessor {index}: {arr.shape} != {v.shape}')
        arr[:]=v
    def mesh_primitive(self,name):
        for mi,m in enumerate(self.js['meshes']):
            if m.get('name')==name:return mi,m['primitives'][0]
        raise KeyError(name)
    def save(self,path):
        jsb=json.dumps(self.js,separators=(',',':'),ensure_ascii=False).encode('utf-8')
        jsb+=b' ' *((4-len(jsb)%4)%4);binb=bytes(self.bin);binb+=b'\x00'*((4-len(binb)%4)%4)
        total=12+8+len(jsb)+8+len(binb)
        out=bytearray(struct.pack('<4sII',b'glTF',self.version,total));out+=struct.pack('<II',len(jsb),0x4E4F534A)+jsb;out+=struct.pack('<II',len(binb),0x004E4942)+binb
        Path(path).write_bytes(out)

def compute_tangents(vertices,faces,uv,normals,fallback):
    tan=np.zeros_like(vertices,dtype=np.float64);bit=np.zeros_like(vertices,dtype=np.float64)
    for tri in faces:
        i,j,k=map(int,tri);p0,p1,p2=vertices[[i,j,k]];w0,w1,w2=uv[[i,j,k]];e1=p1-p0;e2=p2-p0;d1=w1-w0;d2=w2-w0;den=d1[0]*d2[1]-d1[1]*d2[0]
        if abs(den)<1e-12:continue
        r=1.0/den;t=(e1*d2[1]-e2*d1[1])*r;b=(e2*d1[0]-e1*d2[0])*r
        tan[[i,j,k]]+=t;bit[[i,j,k]]+=b
    n=normals.astype(np.float64);t=tan-n*np.sum(n*tan,axis=1,keepdims=True);ln=np.linalg.norm(t,axis=1);bad=ln<1e-10
    t[~bad]/=ln[~bad,None];t[bad]=fallback[bad,:3]
    s=np.where(np.sum(np.cross(n,t)*bit,axis=1)<0,-1.0,1.0)
    out=np.column_stack([t,s]).astype(np.float32);out[bad,3]=fallback[bad,3]
    return out

def patch_outfit(source_path,output_path,positions_by_mesh,hide_body=True):
    import trimesh
    ed=GLBEditor(source_path)
    for name,newpos in positions_by_mesh.items():
        mi,pr=ed.mesh_primitive(name);pos_acc=pr['attributes']['POSITION'];old=ed.accessor(pos_acc);new=np.asarray(newpos,dtype=np.float32)
        if old.shape!=new.shape:raise ValueError(f'{name} position count mismatch {old.shape} vs {new.shape}')
        idx=ed.accessor(pr['indices']).reshape(-1).astype(np.int64);faces=idx.reshape(-1,3)
        mesh=trimesh.Trimesh(vertices=new,faces=faces,process=False);normals=np.asarray(mesh.vertex_normals,dtype=np.float32)
        ed.write_accessor(pos_acc,new);ed.write_accessor(pr['attributes']['NORMAL'],normals)
        if 'TANGENT' in pr['attributes'] and 'TEXCOORD_0' in pr['attributes']:
            uv=ed.accessor(pr['attributes']['TEXCOORD_0']).astype(np.float32);fallback=ed.accessor(pr['attributes']['TANGENT']).astype(np.float32)
            ed.write_accessor(pr['attributes']['TANGENT'],compute_tangents(new,faces,uv,normals,fallback))
        ed.js['accessors'][pos_acc]['min']=new.min(axis=0).astype(float).tolist();ed.js['accessors'][pos_acc]['max']=new.max(axis=0).astype(float).tolist()
    if hide_body:
        body_mesh_indices={i for i,m in enumerate(ed.js.get('meshes',[])) if m.get('name','').startswith('mesh 0.')}
        for scene in ed.js.get('scenes',[]):
            scene['nodes']=[ni for ni in scene.get('nodes',[]) if ed.js['nodes'][ni].get('mesh') not in body_mesh_indices]
    ed.save(output_path)
