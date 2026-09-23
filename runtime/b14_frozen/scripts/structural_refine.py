from __future__ import annotations
import numpy as np, torch, math
from collections import defaultdict

def clean_component_faces(F, ids):
    ids=np.asarray(ids,dtype=np.int64); loc=-np.ones(int(F.max())+1,dtype=np.int64);loc[ids]=np.arange(len(ids))
    mask=np.all(np.isin(F,ids),axis=1); G=loc[F[mask]]
    good=(G[:,0]!=G[:,1])&(G[:,1]!=G[:,2])&(G[:,2]!=G[:,0]);return G[good]

def unique_edges(F):
    return np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)

def cotan_graph(V,F):
    acc=defaultdict(float)
    for tri in F:
        p=V[tri]
        for k,(a,b,c) in enumerate(((1,2,0),(2,0,1),(0,1,2))):
            u=p[a]-p[c];v=p[b]-p[c];den=np.linalg.norm(np.cross(u,v));cot=float(np.dot(u,v)/max(den,1e-12));i,j=sorted((int(tri[a]),int(tri[b])));acc[(i,j)]+=0.5*cot
    e=np.array(list(acc.keys()),dtype=np.int64);w=np.array([acc[tuple(x)] for x in e],dtype=np.float64)
    # Clamp pathological cotangents from skinny triangles, preserving sign only mildly.
    lim=max(np.percentile(np.abs(w),98),1e-3);w=np.clip(w,-lim,lim)
    return e,w

def lap_delta_torch(V,e,w):
    n=V.shape[0];i=e[:,0];j=e[:,1];d=V[j]-V[i];out=torch.zeros_like(V);out.index_add_(0,i,w[:,None]*d);out.index_add_(0,j,-w[:,None]*d);return out

def face_pair_data(V,F):
    owners=defaultdict(list)
    for fi,t in enumerate(F):
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):owners[tuple(sorted((int(a),int(b))))].append(fi)
    pairs=[];weights=[]
    for (a,b),fs in owners.items():
        if len(fs)==2:pairs.append(fs);weights.append(np.linalg.norm(V[a]-V[b]))
    return np.asarray(pairs,dtype=np.int64),np.asarray(weights,dtype=np.float64),owners


def planar_boundary_groups(V,F,threshold=.08,min_vertices=12):
    _,_,owners=face_pair_data(V,F);adj=defaultdict(list)
    for (a,b),fs in owners.items():
        if len(fs)==1:adj[a].append(b);adj[b].append(a)
    seen=set();groups=[]
    for st in adj:
        if st in seen:continue
        stack=[st];seen.add(st);g=[]
        while stack:
            q=stack.pop();g.append(q)
            for nb in adj[q]:
                if nb not in seen:seen.add(nb);stack.append(nb)
        if len(g)<min_vertices:continue
        Q=V[g]-V[g].mean(0);sv=np.linalg.svd(Q,compute_uv=False)/np.sqrt(len(g));ratio=float(sv[-1]/max(sv[-2],1e-12))
        if ratio<threshold:groups.append((np.asarray(g,dtype=np.int64),ratio))
    return groups

def boundary_triplets(V,F):
    _,_,owners=face_pair_data(V,F);adj=defaultdict(list)
    for (a,b),fs in owners.items():
        if len(fs)==1:adj[a].append(b);adj[b].append(a)
    trip=[]
    for i,nbr in adj.items():
        if len(nbr)==2:trip.append((nbr[0],i,nbr[1]))
    return np.asarray(trip,dtype=np.int64),adj

def triangle_shape_signature_torch(V,F):
    a,b,c=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    L=torch.stack([torch.linalg.norm(b-a,dim=1),torch.linalg.norm(c-b,dim=1),torch.linalg.norm(a-c,dim=1)],1)
    return L/(L.sum(1,keepdim=True)+1e-12)

def make_contact_frames(source_body_normals, A):
    # Under y=A x, normals transform by A^-T.
    try: invT=np.linalg.inv(A).transpose(0,2,1)
    except np.linalg.LinAlgError: invT=np.linalg.pinv(A).transpose(0,2,1)
    n=np.einsum('nij,nj->ni',invT,source_body_normals);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    ref=np.tile(np.array([0.,1.,0.]),(len(n),1));parallel=np.abs((ref*n).sum(1))>.9;ref[parallel]=np.array([1.,0.,0.])
    t1=np.cross(n,ref);t1/=np.maximum(np.linalg.norm(t1,axis=1,keepdims=True),1e-12);t2=np.cross(n,t1);t2/=np.maximum(np.linalg.norm(t2,axis=1,keepdims=True),1e-12)
    return t1,t2,n

def refine_component(Vsrc,Vinit,F,Ncontact,iterations=800,lr=.03,tangent_cap=.030,normal_cap=.008,weights=None,log_every=100,triangle_guard_reference=None,triangle_area_floor=.0,triangle_orient_floor=.0,triangle_guard_weight=.0,triangle_guard_start=0):
    weights=weights or {}
    wl=weights.get('lap',8.0);wb=weights.get('bend',1.5);wc=weights.get('curv',1.5);ws=weights.get('shape',2.0);wn=weights.get('normal',2.0);wr=weights.get('reg',0.04);wsm=weights.get('disp_smooth',0.8);wp=weights.get('plane',2.0)
    device='cpu';dtype=torch.float64
    V0=torch.tensor(Vsrc,dtype=dtype,device=device);B=torch.tensor(Vinit,dtype=dtype,device=device);Ft=torch.tensor(F,dtype=torch.long,device=device)
    E_np,Cw_np=cotan_graph(Vsrc,F);E=torch.tensor(E_np,dtype=torch.long);Cw=torch.tensor(Cw_np,dtype=dtype)
    d0=lap_delta_torch(V0,E,Cw);d0=d0/(torch.sqrt(torch.sum(d0*d0))+1e-12)
    pairs_np,pw_np,_=face_pair_data(Vsrc,F);pairs=torch.tensor(pairs_np,dtype=torch.long);pw=torch.tensor(pw_np/max(pw_np.mean() if len(pw_np) else 1,1e-12),dtype=dtype) if len(pw_np) else None
    tri0=triangle_shape_signature_torch(V0,Ft).detach()
    guard_ref=None
    if triangle_guard_reference is not None and triangle_guard_weight>0:
        G0=torch.tensor(np.asarray(triangle_guard_reference,float),dtype=dtype,device=device)
        gc0=torch.linalg.cross(G0[Ft[:,1]]-G0[Ft[:,0]],G0[Ft[:,2]]-G0[Ft[:,0]],dim=1)
        gden=torch.linalg.norm(gc0,dim=1).clamp_min(1e-12);gn0=gc0/gden[:,None];guard_ref=(gden,gn0)
    bt_np,_=boundary_triplets(Vsrc,F);bt=torch.tensor(bt_np,dtype=torch.long) if len(bt_np) else None
    if bt is not None:
        e1=V0[bt[:,0]]-V0[bt[:,1]];e2=V0[bt[:,2]]-V0[bt[:,1]];curv0=torch.sum(torch.nn.functional.normalize(e1,dim=1)*torch.nn.functional.normalize(e2,dim=1),1).detach()
    pgroups=[]
    for g,r in planar_boundary_groups(Vsrc,F):
        qs=Vsrc[g]-Vsrc[g].mean(0);_,_,vhs=np.linalg.svd(qs,full_matrices=False);ns=vhs[-1];dsrc=qs@ns
        pgroups.append((torch.tensor(g,dtype=torch.long),torch.tensor(ns,dtype=dtype),torch.tensor(dsrc,dtype=dtype)))
    N=torch.tensor(Ncontact,dtype=dtype);ref=torch.tensor(np.tile([0.,1.,0.],(len(Vsrc),1)),dtype=dtype);parallel=torch.abs(torch.sum(N*ref,1))>.9;ref[parallel]=torch.tensor([1.,0.,0.],dtype=dtype)
    T1=torch.nn.functional.normalize(torch.linalg.cross(N,ref),dim=1);T2=torch.nn.functional.normalize(torch.linalg.cross(N,T1),dim=1)
    u=torch.zeros((len(Vsrc),3),dtype=dtype,requires_grad=True);opt=torch.optim.AdamW([u],lr=lr,weight_decay=0)
    # Edge graph for correction-field smoothness.
    Ei=E[:,0];Ej=E[:,1]
    hist=[]
    for it in range(iterations+1):
        th=torch.tanh(u);disp=T1*(tangent_cap*th[:,0,None])+T2*(tangent_cap*th[:,1,None])+N*(normal_cap*th[:,2,None]);V=B+disp
        dc=lap_delta_torch(V,E,Cw);dc=dc/(torch.sqrt(torch.sum(dc*dc))+1e-12);Llap=torch.sum((dc-d0)**2)
        # Candidate source-relative bending: compare adjacent face-normal dot products.
        a,b,c=V[Ft[:,0]],V[Ft[:,1]],V[Ft[:,2]];fn=torch.nn.functional.normalize(torch.linalg.cross(b-a,c-a),dim=1)
        a0,b0,c0=V0[Ft[:,0]],V0[Ft[:,1]],V0[Ft[:,2]];fn0=torch.nn.functional.normalize(torch.linalg.cross(b0-a0,c0-a0),dim=1)
        if len(pairs_np):
            cos=torch.sum(fn[pairs[:,0]]*fn[pairs[:,1]],1);cos0=torch.sum(fn0[pairs[:,0]]*fn0[pairs[:,1]],1);Lb=torch.mean(pw*(cos-cos0)**2)
        else: Lb=torch.tensor(0.,dtype=dtype)
        if bt is not None:
            e1=V[bt[:,0]]-V[bt[:,1]];e2=V[bt[:,2]]-V[bt[:,1]];curv=torch.sum(torch.nn.functional.normalize(e1,dim=1)*torch.nn.functional.normalize(e2,dim=1),1);Lc=torch.mean((curv-curv0)**2)
        else:Lc=torch.tensor(0.,dtype=dtype)
        tri=triangle_shape_signature_torch(V,Ft);Ls=torch.mean((tri-tri0)**2)*100.0
        Lguard=torch.tensor(0.,dtype=dtype)
        guard_area_min=guard_orient_min=0.0
        if guard_ref is not None and it>=int(triangle_guard_start):
            gden,gn0=guard_ref;gc=torch.linalg.cross(b-a,c-a,dim=1);gar=torch.linalg.norm(gc,dim=1)/gden;gor=torch.sum(gc*gn0,dim=1)/gden
            ba=torch.relu(float(triangle_area_floor)-gar) if triangle_area_floor>0 else torch.zeros_like(gar)
            bo=torch.relu(float(triangle_orient_floor)-gor) if triangle_orient_floor>0 else torch.zeros_like(gor)
            # Sum rather than mean: this is a sparse barrier and must not vanish simply because the panel has many faces.
            Lguard=torch.sum(ba*ba+bo*bo);guard_area_min=float(gar.min().detach());guard_orient_min=float(gor.min().detach())
        normal_delta=torch.sum((V-B)*N,1);Ln=torch.mean((normal_delta/0.004)**2)
        Lr=torch.mean((disp/0.020)**2)
        Lsm=torch.mean(torch.sum((disp[Ei]-disp[Ej])**2,1))/(0.010**2)
        Lp=torch.tensor(0.,dtype=dtype)
        if pgroups:
            for gg,ns0,dsrc in pgroups:
                qq=V[gg]-torch.mean(V[gg],dim=0,keepdim=True);dcand=qq@ns0;Lp=Lp+torch.mean(((dcand-dsrc)/0.004)**2)
            Lp=Lp/len(pgroups)
        loss=wl*Llap+wb*Lb+wc*Lc+ws*Ls+wn*Ln+wr*Lr+wsm*Lsm+wp*Lp+float(triangle_guard_weight)*Lguard
        if it%log_every==0 or it==iterations:
            hist.append({'iter':it,'loss':float(loss.detach()),'lap':float(Llap.detach()),'bend':float(Lb.detach()),'curv':float(Lc.detach()),'shape':float(Ls.detach()),'normal':float(Ln.detach()),'reg':float(Lr.detach()),'smooth':float(Lsm.detach()),'plane':float(Lp.detach()),'triangle_guard':float(Lguard.detach()),'guard_area_min':guard_area_min,'guard_orient_min':guard_orient_min,'disp_rms_mm':float(torch.sqrt(torch.mean(torch.sum(disp*disp,1))).detach()*1000)})
        if it==iterations:break
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([u],5.0);opt.step()
    return V.detach().cpu().numpy(),hist

def quality(Vsrc,V,F):
    E=unique_edges(F);l0=np.linalg.norm(Vsrc[E[:,0]]-Vsrc[E[:,1]],axis=1);l=np.linalg.norm(V[E[:,0]]-V[E[:,1]],axis=1);rat=l/np.maximum(l0,1e-12)
    A0=np.linalg.norm(np.cross(Vsrc[F[:,1]]-Vsrc[F[:,0]],Vsrc[F[:,2]]-Vsrc[F[:,0]]),axis=1)/2;A=np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)/2;ar=A/np.maximum(A0,1e-12)
    return {'edge_p':np.percentile(rat,[1,5,50,95,99]).tolist(),'area_p':np.percentile(ar,[1,5,50,95,99]).tolist(),'tiny_area_frac':float(np.mean(A<1e-10))}

def component_kind_pca(P, is_root=False):
    if is_root:return 'shell'
    q=P-P.mean(0);sv=np.linalg.svd(q,compute_uv=False)/max(np.sqrt(len(P)),1)
    # Generic narrow deformable strip: long physical span with small thickness.
    if sv[0]>.020 and sv[2]<.0045:return 'ribbon'
    return 'rigid'

def attachment_pairs(V,lab,max_source_gap=.0030,band=.0010,max_per_edge=48):
    from scipy.spatial import cKDTree
    cc=int(lab.max())+1;out=[]
    for a in range(cc):
        ia=np.where(lab==a)[0];Pa=V[ia]
        for b in range(a+1,cc):
            ib=np.where(lab==b)[0];Pb=V[ib];tb=cKDTree(Pb);d,j=tb.query(Pa);mn=float(d.min())
            if mn>max_source_gap:continue
            threshold=min(max_source_gap,mn+band);sel=np.where(d<=threshold)[0]
            if len(sel)>max_per_edge:sel=sel[np.argsort(d[sel])[:max_per_edge]]
            pairs=[(int(ia[k]),int(ib[j[k]]),float(d[k])) for k in sel]
            # add reverse nearests for better coverage
            ta=cKDTree(Pa);dr,jr=ta.query(Pb);selr=np.where(dr<=threshold)[0]
            if len(selr)>max_per_edge:selr=selr[np.argsort(dr[selr])[:max_per_edge]]
            pairs += [(int(ia[jr[k]]),int(ib[k]),float(dr[k])) for k in selr]
            # unique global vertex pairs
            seen=set();uniq=[]
            for x in sorted(pairs,key=lambda z:z[2]):
                key=(x[0],x[1])
                if key not in seen:seen.add(key);uniq.append(x)
            out.append({'a':a,'b':b,'min_source':mn,'pairs':uniq[:max_per_edge]})
    return out

def axis_angle_matrix(r):
    # differentiable Rodrigues, r (...,3)
    theta=torch.linalg.norm(r,dim=-1,keepdim=True);k=r/(theta+1e-12)
    kx,ky,kz=k[...,0],k[...,1],k[...,2];z=torch.zeros_like(kx)
    K=torch.stack([z,-kz,ky,kz,z,-kx,-ky,kx,z],dim=-1).reshape(r.shape[:-1]+(3,3));I=torch.eye(3,dtype=r.dtype,device=r.device).expand(K.shape)
    th=theta[...,0,None,None];return I+torch.sin(th)*K+(1-torch.cos(th))*(K@K)

def refine_assembly(Vsrc,Vinit,F,lab,root_component=0,iterations=800,lr=.025,max_gap=.0030,log_every=100):
    dtype=torch.float64;device='cpu';Vsrc=np.asarray(Vsrc,float);Vinit=np.asarray(Vinit,float);cc=int(lab.max())+1
    comp_ids=[np.where(lab==c)[0] for c in range(cc)];kinds={c:component_kind_pca(Vsrc[comp_ids[c]],c==root_component) for c in range(cc)}
    edges=attachment_pairs(Vsrc,lab,max_source_gap=max_gap)
    # Only optimize components connected (possibly transitively) to root.
    reachable={root_component};changed=True
    while changed:
        changed=False
        for e in edges:
            if e['a'] in reachable and e['b'] not in reachable:reachable.add(e['b']);changed=True
            if e['b'] in reachable and e['a'] not in reachable:reachable.add(e['a']);changed=True
    Vs=torch.tensor(Vsrc,dtype=dtype);B=torch.tensor(Vinit,dtype=dtype);Vroot=B.clone().detach();frame_targets=root_attachment_targets(Vsrc,Vinit,F,lab,root_component,edges)
    # Parameters per rigid component and per ribbon vertex.
    rigid_params={};ribbon_params={};rigid_base={}
    for c in sorted(reachable):
        if c==root_component:continue
        ids=comp_ids[c]
        if kinds[c]=='rigid':
            R0,cs,cd=_rigid_fit_np(Vsrc[ids],Vinit[ids]);rigid_base[c]=(torch.tensor(R0,dtype=dtype),torch.tensor(cs,dtype=dtype),torch.tensor(cd,dtype=dtype));rigid_params[c]=torch.zeros(6,dtype=dtype,requires_grad=True)
        else:ribbon_params[c]=torch.zeros((len(ids),3),dtype=dtype,requires_grad=True)
    params=list(rigid_params.values())+list(ribbon_params.values());opt=torch.optim.AdamW(params,lr=lr,weight_decay=0)
    # ribbon source shape caches
    rcache={}
    for c,p in ribbon_params.items():
        ids=comp_ids[c];G=clean_component_faces(F,ids);E_np,cw_np=cotan_graph(Vsrc[ids],G);E=torch.tensor(E_np,dtype=torch.long);cw=torch.tensor(cw_np,dtype=dtype);v0=Vs[torch.tensor(ids,dtype=torch.long)];d0=lap_delta_torch(v0,E,cw);d0=d0/(torch.sqrt(torch.sum(d0*d0))+1e-12);tri0=triangle_shape_signature_torch(v0,torch.tensor(G,dtype=torch.long)).detach();rcache[c]=(torch.tensor(ids,dtype=torch.long),torch.tensor(G,dtype=torch.long),E,cw,d0,tri0)
    hist=[]
    def buildV():
        chunks=[];V=B.clone();
        for c,p in rigid_params.items():
            ids=torch.tensor(comp_ids[c],dtype=torch.long);R0,cs,cd=rigid_base[c];dr=.65*torch.tanh(p[:3]);dt=.030*torch.tanh(p[3:]);Rd=axis_angle_matrix(dr[None,:])[0];R=Rd@R0;V[ids]=(Vs[ids]-cs)@R.T+cd+dt
        for c,p in ribbon_params.items():
            ids=torch.tensor(comp_ids[c],dtype=torch.long);V[ids]=B[ids]+.025*torch.tanh(p)
        V[torch.tensor(comp_ids[root_component],dtype=torch.long)]=Vroot[torch.tensor(comp_ids[root_component],dtype=torch.long)]
        return V
    for it in range(iterations+1):
        V=buildV();La=torch.tensor(0.,dtype=dtype);npairs=0
        for e in edges:
            if e['a'] not in reachable or e['b'] not in reachable:continue
            arr=e['pairs'];ia=torch.tensor([x[0] for x in arr],dtype=torch.long);ib=torch.tensor([x[1] for x in arr],dtype=torch.long);d0=torch.tensor([x[2] for x in arr],dtype=dtype);d=torch.linalg.norm(V[ia]-V[ib],dim=1);La+=torch.mean(((d-d0)/.0015)**2);npairs+=1
        if npairs:La/=npairs
        Lframe=torch.tensor(0.,dtype=dtype)
        if frame_targets:
            ci=torch.tensor([x[0] for x in frame_targets],dtype=torch.long);des=torch.tensor(np.asarray([x[2] for x in frame_targets]),dtype=dtype);Lframe=torch.mean(torch.sum((V[ci]-des)**2,1))/(.0025**2)
        Lshape=torch.tensor(0.,dtype=dtype);nr=0
        for c,(ids,G,E,cw,d0,tri0) in rcache.items():
            vv=V[ids];dc=lap_delta_torch(vv,E,cw);dc=dc/(torch.sqrt(torch.sum(dc*dc))+1e-12);ll=torch.sum((dc-d0)**2);tri=triangle_shape_signature_torch(vv,G);ls=torch.mean((tri-tri0)**2)*100;Lshape+=6*ll+2*ls;nr+=1
        if nr:Lshape/=nr
        # Weak stay-near initialization; lets attachments dominate but avoids graph drift.
        nonroot=np.where(lab!=root_component)[0];idxnr=torch.tensor(nonroot,dtype=torch.long);Lreg=torch.mean(torch.sum((V[idxnr]-B[idxnr])**2,1))/(.015**2)
        loss=8.0*La+5.0*Lframe+Lshape+.05*Lreg
        if it%log_every==0 or it==iterations:
            gaps=[]
            Vn=V.detach().cpu().numpy()
            for e in edges:
                arr=e['pairs'];ds=[np.linalg.norm(Vn[x[0]]-Vn[x[1]]) for x in arr];gaps.append({'a':e['a'],'b':e['b'],'source_min_mm':e['min_source']*1000,'cand_pair_med_mm':float(np.median(ds)*1000),'cand_pair_p95_mm':float(np.percentile(ds,95)*1000)})
            hist.append({'iter':it,'loss':float(loss.detach()),'attach':float(La.detach()),'frame':float(Lframe.detach()),'shape':float(Lshape.detach()),'reg':float(Lreg.detach()),'gaps':gaps})
        if it==iterations:break
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,10.0);opt.step()
    return buildV().detach().cpu().numpy(),kinds,edges,hist

def _rigid_fit_np(src,dst):
    w=np.ones(len(src))/len(src);cs=(src*w[:,None]).sum(0);cd=(dst*w[:,None]).sum(0);H=((src-cs)*w[:,None]).T@(dst-cd);u,s,vt=np.linalg.svd(H);R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    return R,cs,cd

def vertex_normals_np(V,F):
    n=np.zeros_like(V);fn=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
    for k in range(3):np.add.at(n,F[:,k],fn)
    n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12);return n

def root_attachment_targets(Vsrc,Vroot,F,lab,root_component,edges):
    ids=np.where(lab==root_component)[0];g2l={int(g):i for i,g in enumerate(ids)};G=clean_component_faces(F,ids);S=Vsrc[ids];T=Vroot[ids]
    ns=vertex_normals_np(S,G);nt=vertex_normals_np(T,G);adj=[[] for _ in range(len(ids))]
    for a,b in unique_edges(G):adj[a].append(b);adj[b].append(a)
    R=[]
    for i in range(len(ids)):
        if adj[i]:
            # longest incident source edge gives a stable split-vertex-independent tangent.
            j=max(adj[i],key=lambda q:np.linalg.norm(S[q]-S[i]));ts=S[j]-S[i];tt=T[j]-T[i]
        else:ts=np.array([1.,0.,0.]);tt=ts.copy()
        ts=ts-ns[i]*np.dot(ts,ns[i]);tt=tt-nt[i]*np.dot(tt,nt[i]);
        if np.linalg.norm(ts)<1e-8:ts=np.cross(ns[i],[0,1,0] if abs(ns[i,1])<.9 else [1,0,0])
        if np.linalg.norm(tt)<1e-8:tt=np.cross(nt[i],[0,1,0] if abs(nt[i,1])<.9 else [1,0,0])
        ts/=np.linalg.norm(ts);tt/=np.linalg.norm(tt);bs=np.cross(ns[i],ts);bt=np.cross(nt[i],tt);Bs=np.stack([ts,bs,ns[i]],1);Bt=np.stack([tt,bt,nt[i]],1);R.append(Bt@Bs.T)
    R=np.asarray(R);out=[]
    for e in edges:
        if e['a']!=root_component and e['b']!=root_component:continue
        for ia,ib,d in e['pairs']:
            if e['a']==root_component:rglob,cglob=ia,ib
            else:rglob,cglob=ib,ia
            if rglob not in g2l:continue
            r=g2l[rglob];off=Vsrc[cglob]-Vsrc[rglob];desired=Vroot[rglob]+R[r]@off;out.append((cglob,rglob,desired,e['a'],e['b']))
    return out
