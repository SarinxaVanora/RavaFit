from __future__ import annotations
import json, struct, math, hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

COMP={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
NCOMP={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}

class GLB:
    def __init__(self,path):
        self.path=str(path); b=Path(path).read_bytes(); magic,version,length=struct.unpack_from('<4sII',b,0)
        if magic!=b'glTF': raise ValueError(path)
        self.version=version; off=12; chunks=[]
        while off<length:
            clen,ctype=struct.unpack_from('<II',b,off); off+=8; data=b[off:off+clen]; off+=clen; chunks.append((ctype,data))
        self.js=json.loads(chunks[0][1].decode('utf8').rstrip('\x00 ')); self.bin=bytearray(next(d for t,d in chunks if t==0x004E4942))
        self._mesh_nodes={}
        for ni,n in enumerate(self.js.get('nodes',[])):
            if 'mesh' in n:self._mesh_nodes[n['mesh']]=ni
    def accessor(self,i,copy=True):
        a=self.js['accessors'][i]; bv=self.js['bufferViews'][a['bufferView']]; dt=np.dtype(COMP[a['componentType']]).newbyteorder('<'); nc=NCOMP[a['type']]
        off=bv.get('byteOffset',0)+a.get('byteOffset',0); stride=bv.get('byteStride',dt.itemsize*nc)
        arr=np.ndarray((a['count'],nc),dtype=dt,buffer=self.bin,offset=off,strides=(stride,dt.itemsize))
        return arr.copy() if copy else arr
    def primitive(self,name):
        for mi,m in enumerate(self.js['meshes']):
            if m.get('name')==name:return mi,m['primitives'][0]
        raise KeyError(name)
    def mesh_names(self):return [m.get('name') for m in self.js.get('meshes',[])]
    def material_name(self,p):
        i=p.get('material'); return '' if i is None else self.js.get('materials',[{}])[i].get('name','')
    def data(self,name):
        mi,p=self.primitive(name); a=p['attributes']; V=self.accessor(a['POSITION']).astype(np.float64); F=self.accessor(p['indices']).reshape(-1,3).astype(np.int64)
        UV=self.accessor(a['TEXCOORD_0']).astype(np.float64) if 'TEXCOORD_0' in a else None
        N=self.accessor(a['NORMAL']).astype(np.float64) if 'NORMAL' in a else None
        node=self.js['nodes'][self._mesh_nodes[mi]]; skin=self.js['skins'][node['skin']]
        names=[self.js['nodes'][ni].get('name','') for ni in skin['joints']]
        W=np.zeros((len(V),len(names)),np.float64)
        for suf in ['0','1']:
            jk='JOINTS_'+suf; wk='WEIGHTS_'+suf
            if jk not in a:continue
            J=self.accessor(a[jk]).astype(np.int64); WW=self.accessor(a[wk]).astype(np.float64)
            for c in range(J.shape[1]): W[np.arange(len(V)),J[:,c]]+=WW[:,c]
        sw=W.sum(1); good=sw>1e-12; W[good]/=sw[good,None]
        return {'name':name,'mi':mi,'p':p,'V':V,'F':F,'UV':UV,'N':N,'W':W,'joint_names':names,'material':self.material_name(p)}
    def write_accessor(self,i,values):
        arr=self.accessor(i,False); v=np.asarray(values,dtype=arr.dtype)
        if arr.shape!=v.shape:raise ValueError((i,arr.shape,v.shape));arr[:]=v
    def save(self,path):
        jsb=json.dumps(self.js,separators=(',',':'),ensure_ascii=False).encode();jsb+=b' '*((4-len(jsb)%4)%4);bb=bytes(self.bin);bb+=b'\0'*((4-len(bb)%4)%4)
        out=bytearray(struct.pack('<4sII',b'glTF',self.version,12+8+len(jsb)+8+len(bb)));out+=struct.pack('<II',len(jsb),0x4E4F534A)+jsb;out+=struct.pack('<II',len(bb),0x004E4942)+bb;Path(path).write_bytes(out)

def weld_mesh(d,eps=1e-7):
    V=d['V']; key=np.round(V/eps).astype(np.int64); _,first,inv=np.unique(key,axis=0,return_index=True,return_inverse=True)
    order=np.argsort(first); # np.unique lexicographic order is fine; retain inv mapping accordingly rather than reorder
    # leave unique in lexicographic key order for stable mapping
    Ukey,first,inv=np.unique(key,axis=0,return_index=True,return_inverse=True); U=V[first]
    F=inv[d['F']]
    W=np.zeros((len(U),d['W'].shape[1]),float); counts=np.zeros(len(U),float)
    np.add.at(W,inv,d['W']); np.add.at(counts,inv,1); W/=np.maximum(counts[:,None],1); W/=np.maximum(W.sum(1,keepdims=True),1e-12)
    UV=None
    if d['UV'] is not None:
        UV=np.zeros((len(U),2),float);np.add.at(UV,inv,d['UV']);UV/=np.maximum(counts[:,None],1)
    return {'V':U,'F':F,'W':W,'UV':UV,'raw_to_weld':inv,'raw_V':V,'raw_F':d['F'],'joint_names':d['joint_names'],'material':d['material'],'name':d['name']}

def connected_labels(w):
    F=w['F']; n=len(w['V']);i=np.r_[F[:,0],F[:,1],F[:,2],F[:,0],F[:,1],F[:,2]];j=np.r_[F[:,1],F[:,2],F[:,0],F[:,2],F[:,0],F[:,1]]
    A=coo_matrix((np.ones(len(i)),(i,j)),shape=(n,n)).tocsr();A.data[:]=1
    return connected_components(A,directed=False)

def uv_map_vertices(src,tgt,k=12):
    tree=cKDTree(tgt['UV']); dist,idx=tree.query(src['UV'],k=min(k,len(tgt['V'])))
    if dist.ndim==1:dist=dist[:,None];idx=idx[:,None]
    ww=1/(dist*dist+1e-7);ww/=ww.sum(1,keepdims=True)
    P=(tgt['V'][idx]*ww[:,:,None]).sum(1)
    W=(tgt['W'][idx]*ww[:,:,None]).sum(1)
    return P,W,np.min(dist,axis=1)

def collect_body_pairs(src_glb,tgt_glb):
    X=[];Y=[];W=[]; names=None;stats={}
    for name in src_glb.mesh_names():
        if not name or not name.startswith('mesh 0.'):continue
        sd=src_glb.data(name)
        if 'bibo' not in sd['material'].lower():continue
        if name not in tgt_glb.mesh_names():continue
        td=tgt_glb.data(name)
        if 'bibo' not in td['material'].lower():continue
        if names is None:names=sd['joint_names']
        if sd['joint_names']!=names:raise ValueError('source body skin joint order changed')
        if td['joint_names']!=names:raise ValueError('target body skin joint order mismatch')
        yp,tw,uvd=uv_map_vertices(sd,td)
        X.append(sd['V']);Y.append(yp);W.append(sd['W'])
        stats[name]={'n':len(sd['V']),'uv_p50':float(np.median(uvd)),'uv_p95':float(np.percentile(uvd,95)),'disp_rms_mm':float(np.sqrt(np.mean(np.sum((yp-sd['V'])**2,axis=1)))*1000)}
    return np.vstack(X),np.vstack(Y),np.vstack(W),names,stats

def fit_bone_affines(X,Y,W,names,weight_floor=0.005,ridge=0.025,min_points=25):
    aff=[];stats={}
    for b,name in enumerate(names):
        wb=W[:,b]; ids=wb>weight_floor
        if ids.sum()<min_points or wb[ids].sum()<2:
            aff.append((np.eye(3),np.zeros(3),np.zeros(3)));stats[name]={'mode':'identity','n':int(ids.sum())};continue
        xx=X[ids];yy=Y[ids];wt=wb[ids]**2;ws=wt.sum();cx=(xx*wt[:,None]).sum(0)/ws;cy=(yy*wt[:,None]).sum(0)/ws
        xc=xx-cx;yc=yy-cy;Cxx=(xc*wt[:,None]).T@xc/ws;Cyx=(yc*wt[:,None]).T@xc/ws;lam=ridge*max(np.trace(Cxx)/3,1e-8)
        A=(Cyx+lam*np.eye(3))@np.linalg.inv(Cxx+lam*np.eye(3))
        # prevent pathological poorly-supported transforms; retain legitimate breast shrink
        u,s,vt=np.linalg.svd(A);s=np.clip(s,0.28,1.35);A=u@np.diag(s)@vt
        pred=(xx-cx)@A.T+cy;err=np.linalg.norm(pred-yy,axis=1)
        aff.append((A,cx,cy));stats[name]={'mode':'affine','n':int(ids.sum()),'mass':float(wb.sum()),'singular_values':s.tolist(),'fit_rms_mm':float(np.sqrt(np.average(err*err,weights=wt))*1000)}
    return aff,stats

def remap_weights(W,src_names,common_names):
    out=np.zeros((len(W),len(common_names)),float);idx={n:i for i,n in enumerate(common_names)}
    for j,n in enumerate(src_names):
        if n in idx:out[:,idx[n]]+=W[:,j]
    out/=np.maximum(out.sum(1,keepdims=True),1e-12);return out

def blend_affine_map(P,W,names,affines,topk=8):
    if W.shape[1]!=len(names):raise ValueError('weights/names')
    if topk is not None and topk<W.shape[1]:
        ix=np.argpartition(W,-topk,axis=1)[:,-topk:];mask=np.zeros_like(W,bool);mask[np.arange(len(W))[:,None],ix]=True;W=np.where(mask,W,0);W/=np.maximum(W.sum(1,keepdims=True),1e-12)
    out=np.zeros_like(P)
    active=np.where(W.sum(0)>1e-8)[0]
    for b in active:
        A,cx,cy=affines[b];mapped=(P-cx)@A.T+cy;out+=W[:,b,None]*mapped
    return out

def rigid_fit(src,dst,weights=None):
    if weights is None:weights=np.ones(len(src))
    w=np.maximum(weights,1e-9);w=w/w.sum();cs=(src*w[:,None]).sum(0);cd=(dst*w[:,None]).sum(0);H=((src-cs)*w[:,None]).T@(dst-cd);u,s,vt=np.linalg.svd(H);R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    return R,cs,cd

def classify_components(w):
    cc,lab=connected_labels(w);classes={};details=[]
    for c in range(cc):
        ids=np.where(lab==c)[0];P=w['V'][ids];ext=P.max(0)-P.min(0);ss=np.sort(ext);aspect=ss[-1]/max(ss[0],1e-6)
        # generic geometry-only classifier
        if len(ids)>=300 or (ss[1]>0.035 and ss[-1]>0.06):kind='shell'
        elif ss[-1]>0.055 and ss[0]<0.018:kind='ribbon'
        else:kind='rigid'
        classes[c]=kind;details.append({'component':c,'n':len(ids),'kind':kind,'centroid':P.mean(0).tolist(),'extent':ext.tolist()})
    return lab,classes,details

def map_garment_welded(w,common_names,affines):
    W=remap_weights(w['W'],w['joint_names'],common_names);target=blend_affine_map(w['V'],W,common_names,affines)
    lab,classes,details=classify_components(w);out=target.copy()
    for c,kind in classes.items():
        ids=np.where(lab==c)[0]
        if kind=='rigid' and len(ids)>=3:
            R,cs,cd=rigid_fit(w['V'][ids],target[ids],np.max(W[ids],axis=1)+.1);out[ids]=(w['V'][ids]-cs)@R.T+cd
    return out,lab,classes,details

def expand_welded(w,U):return U[w['raw_to_weld']]

def chamfer(A,B):
    da=cKDTree(B).query(A)[0];db=cKDTree(A).query(B)[0];return {'rms_sym_mm':float(np.sqrt((np.mean(da*da)+np.mean(db*db))/2)*1000),'rms_ab_mm':float(np.sqrt(np.mean(da*da))*1000),'median_ab_mm':float(np.median(da)*1000),'p95_ab_mm':float(np.percentile(da,95)*1000)}

def patch_positions(src_path,out_path,pos_by_mesh,remove_skin_material_body=True):
    from glb_patch_legacy import GLBEditor,compute_tangents
    ed=GLBEditor(src_path)
    for name,new in pos_by_mesh.items():
        mi,p=ed.mesh_primitive(name);attrs=p['attributes'];V=np.asarray(new,np.float32);ed.write_accessor(attrs['POSITION'],V);idx=ed.accessor(p['indices']).reshape(-1).astype(int);F=idx.reshape(-1,3);m=trimesh.Trimesh(V,F,process=False);N=np.asarray(m.vertex_normals,np.float32);ed.write_accessor(attrs['NORMAL'],N)
        if 'TANGENT' in attrs and 'TEXCOORD_0' in attrs:ed.write_accessor(attrs['TANGENT'],compute_tangents(V,F,ed.accessor(attrs['TEXCOORD_0']).astype(np.float32),N,ed.accessor(attrs['TANGENT']).astype(np.float32)))
        ai=attrs['POSITION'];ed.js['accessors'][ai]['min']=V.min(0).astype(float).tolist();ed.js['accessors'][ai]['max']=V.max(0).astype(float).tolist()
    if remove_skin_material_body:
        mats=ed.js.get('materials',[]);body_mesh=set()
        for mi,m in enumerate(ed.js.get('meshes',[])):
            for p in m.get('primitives',[]):
                mat=p.get('material');mn=mats[mat].get('name','').lower() if mat is not None else ''
                if 'bibo' in mn or ('skin' in mn and 'outfit' not in mn):body_mesh.add(mi)
        for scene in ed.js.get('scenes',[]):scene['nodes']=[ni for ni in scene.get('nodes',[]) if ed.js['nodes'][ni].get('mesh') not in body_mesh]
    ed.save(out_path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def collect_body_pairs_full(src_glb,tgt_glb):
    X=[];Y=[];W=[];NS=[];NT=[];names=None;parts=[]
    for name in src_glb.mesh_names():
        if not name or not name.startswith('mesh 0.'):continue
        sd=src_glb.data(name)
        if 'bibo' not in sd['material'].lower() or name not in tgt_glb.mesh_names():continue
        td=tgt_glb.data(name)
        if 'bibo' not in td['material'].lower():continue
        if names is None:names=sd['joint_names']
        yp,tw,uvd=uv_map_vertices(sd,td)
        tree=cKDTree(td['UV']);dist,idx=tree.query(sd['UV'],k=min(12,len(td['V'])));dist=dist if dist.ndim>1 else dist[:,None];idx=idx if idx.ndim>1 else idx[:,None]
        ww=1/(dist*dist+1e-7);ww/=ww.sum(1,keepdims=True)
        nt=(td['N'][idx]*ww[:,:,None]).sum(1);nt/=np.maximum(np.linalg.norm(nt,axis=1,keepdims=True),1e-12)
        ns=sd['N'].copy();ns/=np.maximum(np.linalg.norm(ns,axis=1,keepdims=True),1e-12)
        X.append(sd['V']);Y.append(yp);W.append(sd['W']);NS.append(ns);NT.append(nt);parts.extend([name]*len(sd['V']))
    return np.vstack(X),np.vstack(Y),np.vstack(W),np.vstack(NS),np.vstack(NT),names,np.array(parts,dtype=object)

def body_guided_contacts(P,Wg,X,Y,BW,NS,NT,k=24,bone_penalty=.012):
    tree=cKDTree(X);dist,idx=tree.query(P,k=min(k,len(X)));dist=dist if dist.ndim>1 else dist[:,None];idx=idx if idx.ndim>1 else idx[:,None]
    # cosine-like aligned skin weights; both normalized to sum 1
    align=np.einsum('nk,nqk->nq',Wg,BW[idx]);score=dist+bone_penalty*(1-align)
    pick=np.argmin(score,axis=1);ii=idx[np.arange(len(P)),pick]
    xc=X[ii];yc=Y[ii];ns=NS[ii];nt=NT[ii]
    signed=np.sum((P-xc)*ns,axis=1);eu=np.linalg.norm(P-xc,axis=1)
    return {'source_body_id':ii,'source_point':xc,'target_point':yc,'source_normal':ns,'target_normal':nt,'source_signed':signed,'source_euclid':eu,'score':score[np.arange(len(P)),pick]}

def precompute_body_local_affines(X,Y,BW,k=36,ridge=.04,sv_min=.32,sv_max=1.28):
    tree=cKDTree(X);dist,idx=tree.query(X,k=min(k,len(X))); Aall=np.empty((len(X),3,3),float);quality=np.empty(len(X),float)
    for i in range(len(X)):
        ids=idx[i];xx=X[ids]-X[i];yy=Y[ids]-Y[i]
        sig=max(np.median(dist[i][1:])*2.5,0.004)
        align=BW[ids]@BW[i]
        wt=np.exp(-(dist[i]/sig)**2)*(0.2+align)**2; wt[0]+=3.0; ws=wt.sum()
        Cxx=(xx*wt[:,None]).T@xx/ws; Cyx=(yy*wt[:,None]).T@xx/ws; lam=ridge*max(np.trace(Cxx)/3,1e-9)
        A=(Cyx+lam*np.eye(3))@np.linalg.inv(Cxx+lam*np.eye(3))
        u,s,vt=np.linalg.svd(A);s=np.clip(s,sv_min,sv_max);A=u@np.diag(s)@vt;Aall[i]=A
        pred=xx@A.T;quality[i]=np.sqrt(np.average(np.sum((pred-yy)**2,axis=1),weights=wt))
    return Aall,quality,tree

def body_guided_contact_ids(P,Wg,X,BW,tree=None,k=24,bone_penalty=.015,side_penalty=.06):
    if tree is None:tree=cKDTree(X)
    dist,idx=tree.query(P,k=min(k,len(X)));dist=dist if dist.ndim>1 else dist[:,None];idx=idx if idx.ndim>1 else idx[:,None]
    align=np.einsum('nk,nqk->nq',Wg,BW[idx]);score=dist+bone_penalty*(1-align)
    # strong but generic left/right consistency away from centreline
    sx=np.sign(P[:,0])[:,None];bx=np.sign(X[idx][:,:,0]);score+=np.where((np.abs(P[:,0,None])>.025)&(sx!=bx),side_penalty,0)
    pick=np.argmin(score,axis=1);return idx[np.arange(len(P)),pick],dist[np.arange(len(P)),pick],align[np.arange(len(P)),pick]

def local_body_field_map(P,Wg,X,Y,BW,Aall,tree=None):
    ids,dist,align=body_guided_contact_ids(P,Wg,X,BW,tree=tree)
    out=Y[ids]+np.einsum('nij,nj->ni',Aall[ids],P-X[ids])
    return out,ids,dist,align

def map_garment_local_field(w,common_names,X,Y,BW,Aall,tree):
    W=remap_weights(w['W'],w['joint_names'],common_names);target,contact,dist,align=local_body_field_map(w['V'],W,X,Y,BW,Aall,tree)
    lab,classes,details=classify_components(w);out=target.copy()
    for c,kind in classes.items():
        ids=np.where(lab==c)[0]
        if kind=='rigid' and len(ids)>=3:
            R,cs,cd=rigid_fit(w['V'][ids],target[ids],np.max(W[ids],axis=1)+.1);out[ids]=(w['V'][ids]-cs)@R.T+cd
    return out,lab,classes,details,contact,dist,align

def skeleton_global_positions(glb, joint_names):
    parents={}
    for i,n in enumerate(glb.js.get('nodes',[])):
        for c in n.get('children',[]):parents[c]=i
    def local(n):
        if 'matrix' in n:return np.array(n['matrix'],float).reshape(4,4).T
        t=np.array(n.get('translation',[0,0,0]),float);q=np.array(n.get('rotation',[0,0,0,1]),float);sc=np.array(n.get('scale',[1,1,1]),float);x,y,z,w=q
        R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],float);M=np.eye(4);M[:3,:3]=R@np.diag(sc);M[:3,3]=t;return M
    cache={}
    def glob(i):
        if i in cache:return cache[i]
        M=local(glb.js['nodes'][i]);M=glob(parents[i])@M if i in parents else M;cache[i]=M;return M
    # Resolve first skin whose joint-name order matches.
    nodes_by_name={n.get('name',''):i for i,n in enumerate(glb.js.get('nodes',[]))}
    pos=np.zeros((len(joint_names),3),float);nodeids=[]
    for k,nm in enumerate(joint_names):
        ni=nodes_by_name.get(nm);nodeids.append(ni);pos[k]=glob(ni)[:3,3] if ni is not None else 0
    return pos,nodeids,parents

def skeleton_bone_axes(glb,joint_names):
    pos,nodeids,parents=skeleton_global_positions(glb,joint_names);name_to_idx={n:i for i,n in enumerate(joint_names)};node_to_idx={ni:i for i,ni in enumerate(nodeids) if ni is not None};axes=np.zeros_like(pos)
    for i,ni in enumerate(nodeids):
        vec=None
        if ni is not None and ni in parents and parents[ni] in node_to_idx:
            p=node_to_idx[parents[ni]];v=pos[i]-pos[p]
            if np.linalg.norm(v)>1e-5:vec=v
        if vec is None and ni is not None:
            for ch in glb.js['nodes'][ni].get('children',[]):
                if ch in node_to_idx:
                    v=pos[node_to_idx[ch]]-pos[i]
                    if np.linalg.norm(v)>1e-5:vec=v;break
        if vec is None:vec=np.array([0.,1.,0.])
        axes[i]=vec/np.linalg.norm(vec)
    return axes

def suppress_bone_axis_drift(P,U,W,axes,strength=.85):
    dom=np.argmax(W,axis=1);a=axes[dom];d=U-P;ax=np.sum(d*a,axis=1);return U-strength*ax[:,None]*a

def local_body_field_map_soft(P,Wg,X,Y,BW,Aall,tree=None,k=32,bone_penalty=.015,side_penalty=.08,tau=.0035,max_terms=12):
    """Smooth FFXIV anatomy-aware source->target body field.

    Unlike the hard contact initializer, blend several nearby source-body patches that agree
    with the garment vertex's skin semantics. This preserves anatomical identity without
    imprinting source body tessellation/contact jumps into garment boundaries.
    """
    if tree is None:tree=cKDTree(X)
    dist,idx=tree.query(P,k=min(k,len(X)));dist=dist if dist.ndim>1 else dist[:,None];idx=idx if idx.ndim>1 else idx[:,None]
    align=np.einsum('nk,nqk->nq',Wg,BW[idx]);score=dist+bone_penalty*(1-align)
    sx=np.sign(P[:,0])[:,None];bx=np.sign(X[idx][:,:,0]);score+=np.where((np.abs(P[:,0,None])>.020)&(sx!=bx),side_penalty,0)
    # Keep only the best few anatomically plausible samples before soft blending.
    m=min(max_terms,score.shape[1]);ordr=np.argpartition(score,m-1,axis=1)[:,:m];selidx=np.take_along_axis(idx,ordr,axis=1);sels=np.take_along_axis(score,ordr,axis=1);seld=np.take_along_axis(dist,ordr,axis=1);sela=np.take_along_axis(align,ordr,axis=1)
    rel=sels-sels.min(1,keepdims=True);ww=np.exp(-rel/max(tau,1e-6));ww/=np.maximum(ww.sum(1,keepdims=True),1e-12)
    Xi=X[selidx];Yi=Y[selidx];Ai=Aall[selidx]
    off=P[:,None,:]-Xi;mapped=Yi+np.einsum('nqij,nqj->nqi',Ai,off);out=np.sum(mapped*ww[:,:,None],axis=1)
    # Representative id is retained only for diagnostics/contact-frame lookup.
    best=np.argmin(sels,axis=1);rid=selidx[np.arange(len(P)),best]
    return out,rid,np.sum(seld*ww,axis=1),np.sum(sela*ww,axis=1),ww,selidx

def map_garment_local_field_soft(w,common_names,X,Y,BW,Aall,tree,tau=.0035):
    W=remap_weights(w['W'],w['joint_names'],common_names);target,contact,dist,align,blend,blend_ids=local_body_field_map_soft(w['V'],W,X,Y,BW,Aall,tree=tree,tau=tau)
    lab,classes,details=classify_components(w);out=target.copy()
    for c,kind in classes.items():
        ids=np.where(lab==c)[0]
        if kind=='rigid' and len(ids)>=3:
            R,cs,cd=rigid_fit(w['V'][ids],target[ids],np.max(W[ids],axis=1)+.1);out[ids]=(w['V'][ids]-cs)@R.T+cd
    return out,lab,classes,details,contact,dist,align

def solver_body_weights(P,X,BW,tree=None):
    if tree is None:tree=cKDTree(X)
    d,i=tree.query(P,k=1);W=BW[i].copy();W/=np.maximum(W.sum(1,keepdims=True),1e-12)
    return W,d
