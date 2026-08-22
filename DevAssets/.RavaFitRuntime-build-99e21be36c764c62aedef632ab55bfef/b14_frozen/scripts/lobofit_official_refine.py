from __future__ import annotations
import numpy as np, torch
from structural_refine import cotan_graph, lap_delta_torch, face_pair_data, boundary_triplets
from collision_eval import nearest_surface


def _bone_frames(glb, joint_names):
    # FFXIV adapter: use the existing rig hierarchy as the semantic frame source.
    from ffxiv_lobofit import skeleton_global_positions
    P,nodeids,parents=skeleton_global_positions(glb,joint_names)
    node_to_idx={ni:i for i,ni in enumerate(nodeids) if ni is not None}
    Z=np.zeros_like(P)
    for i,ni in enumerate(nodeids):
        v=None
        # Prefer a child joint: this is the paper's bone head->tail direction.
        if ni is not None:
            for ch in glb.js['nodes'][ni].get('children',[]):
                if ch in node_to_idx:
                    vv=P[node_to_idx[ch]]-P[i]
                    if np.linalg.norm(vv)>1e-7: v=vv;break
        if v is None and ni is not None and ni in parents and parents[ni] in node_to_idx:
            vv=P[i]-P[node_to_idx[parents[ni]]]
            if np.linalg.norm(vv)>1e-7:v=vv
        if v is None:v=np.array([0.,1.,0.])
        Z[i]=v/np.linalg.norm(v)
    X=np.zeros_like(P);Y=np.zeros_like(P)
    # Stable hierarchy-independent orthogonal transverse direction. Since source/target FFXIV
    # rigs are in the same pose this is enough to define a repeatable LoBo frame.
    for i,z in enumerate(Z):
        ref=np.array([1.,0.,0.]) if abs(z[0])<.85 else np.array([0.,0.,1.])
        x=ref-z*np.dot(ref,z)
        if np.linalg.norm(x)<1e-8:x=np.cross([0.,1.,0.],z)
        x/=max(np.linalg.norm(x),1e-12);y=np.cross(z,x);y/=max(np.linalg.norm(y),1e-12);x=np.cross(y,z);x/=max(np.linalg.norm(x),1e-12)
        X[i]=x;Y[i]=y
    # Bone length is only the local-coordinate scale; choose actual child/parent segment lengths.
    L=np.ones(len(P),float)
    for i,ni in enumerate(nodeids):
        vals=[]
        if ni is not None:
            for ch in glb.js['nodes'][ni].get('children',[]):
                if ch in node_to_idx: vals.append(np.linalg.norm(P[node_to_idx[ch]]-P[i]))
        if not vals and ni is not None and ni in parents and parents[ni] in node_to_idx: vals=[np.linalg.norm(P[i]-P[node_to_idx[parents[ni]]])]
        L[i]=max(vals[0] if vals else .05, .005)
    return P,X,Y,Z,L


def _project(P,joints,X,Y,Z,L):
    D=P[:,None,:]-joints[None,:,:]
    return np.stack([np.einsum('nkj,kj->nk',D,X),np.einsum('nkj,kj->nk',D,Y),np.einsum('nkj,kj->nk',D,Z)],2)/L[None,:,None]


def _vertex_area_weights(V,F):
    a=np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)*.5
    w=np.zeros(len(V),float)
    for k in range(3):np.add.at(w,F[:,k],a/3)
    return w/max(w.sum(),1e-12)


def refine_lobofit(Vsrc,Vinit,F,Wfull,joint_names,rig_glb,source_tri,target_tri,iterations=1200,lr=.005,contact_update=200,w_scale=.10,log_every=100,fit_distance=.020,collision_margin=.00035,freeze_axis_strength=1.0,fit_weight=10000.0,collision_weight=10.0,semantic_tangent_weight=0.0,semantic_tangent_scale=.010,lap_weight=.5,bend_weight=1.0,curv_weight=1.0,official_reg_sums=False,collision_area_weighted=False,fit_vertex_mask=None,fit_clearance_scale=None):
    """LoBoFit-style refinement around an FFXIV anatomy-aware initialization.

    The source/control validation mesh is never accepted by this function.  Vinit is produced only
    from source garment + source/target bodies.  Optimization variables and objective mirror the
    official release: bone-local residuals, bounded weight residuals, collision/fit-style,
    normalized Laplacian, bending, boundary curvature, axial and weight regularization.
    """
    Vsrc=np.asarray(Vsrc,float);Vinit=np.asarray(Vinit,float);F=np.asarray(F,np.int64);Wfull=np.asarray(Wfull,float)
    mass=Wfull.sum(0);act=np.where(mass>1e-5)[0]
    W=Wfull[:,act];W/=np.maximum(W.sum(1,keepdims=True),1e-12)
    J,X,Y,Z,L=_bone_frames(rig_glb,joint_names);J,X,Y,Z,L=J[act],X[act],Y[act],Z[act],L[act]
    abk0=_project(Vinit,J,X,Y,Z,L)
    # source fit style, using source only
    _,_,ds,_,_=nearest_surface(Vsrc,source_tri,k=20)
    fitmask=np.abs(ds)<fit_distance if fit_vertex_mask is None else np.asarray(fit_vertex_mask,dtype=bool)
    if fitmask.shape!=(len(Vsrc),):raise ValueError(f'fit_vertex_mask shape {fitmask.shape} != {(len(Vsrc),)}')
    varea=_vertex_area_weights(Vsrc,F)

    dtype=torch.float32;device='cpu'
    Vs=torch.tensor(Vsrc,dtype=dtype);base_abk=torch.tensor(abk0,dtype=dtype);base_w=torch.tensor(W,dtype=dtype)
    tj=torch.tensor(J,dtype=dtype);tx=torch.tensor(X,dtype=dtype);ty=torch.tensor(Y,dtype=dtype);tz=torch.tensor(Z,dtype=dtype);tl=torch.tensor(L,dtype=dtype)
    delta=torch.zeros_like(base_abk,requires_grad=True);dw=torch.zeros_like(base_w,requires_grad=True)
    opt=torch.optim.AdamW([delta,dw],lr=lr,betas=(.9,.999),amsgrad=True,weight_decay=0)
    Ft=torch.tensor(F,dtype=torch.long)
    E_np,cw_np=cotan_graph(Vsrc,F);E=torch.tensor(E_np,dtype=torch.long);cw=torch.tensor(cw_np,dtype=dtype)
    lap0=lap_delta_torch(Vs,E,cw);lap0=lap0/(torch.sqrt(torch.sum(lap0*lap0))+1e-12)
    pairs_np,pw_np,_=face_pair_data(Vsrc,F);pairs=torch.tensor(pairs_np,dtype=torch.long)
    if len(pw_np): pw=torch.tensor(pw_np/max(np.sum(pw_np),1e-12),dtype=dtype)
    else:pw=None
    a0,b0,c0=Vs[Ft[:,0]],Vs[Ft[:,1]],Vs[Ft[:,2]];fn0=torch.nn.functional.normalize(torch.linalg.cross(b0-a0,c0-a0),dim=1)
    bt_np,_=boundary_triplets(Vsrc,F);bt=torch.tensor(bt_np,dtype=torch.long) if len(bt_np) else None
    if bt is not None:
        e1=Vs[bt[:,0]]-Vs[bt[:,1]];e2=Vs[bt[:,2]]-Vs[bt[:,1]];curv0=torch.sum(torch.nn.functional.normalize(e1,dim=1)*torch.nn.functional.normalize(e2,dim=1),1).detach()
        le=(torch.linalg.norm(e1,dim=1)+torch.linalg.norm(e2,dim=1))*.5;cwcur=le/(le.sum()+1e-12)
    fitids=np.where(fitmask)[0];fit_t=torch.tensor(fitids,dtype=torch.long)
    if fit_clearance_scale is None:
        desired_ds=ds
    else:
        sc=np.asarray(fit_clearance_scale,float)
        if sc.shape!=(len(Vsrc),): raise ValueError(f'fit_clearance_scale shape {sc.shape} != {(len(Vsrc),)}')
        desired_ds=ds*sc
    ds_t=torch.tensor(desired_ds[fitids],dtype=dtype);aw=torch.tensor(varea[fitids],dtype=dtype);aw=aw/(aw.sum()+1e-12)
    area_all=torch.tensor(varea,dtype=dtype);area_all=area_all/(area_all.sum()+1e-12)
    # FFXIV-specific semantic anchor: the source/target rigs can be virtually identical even when
    # body volume changes enormously. Vinit is therefore a body-correspondence-derived semantic
    # placement, not just a disposable guess. Penalize only tangential drift from that placement;
    # normal motion remains governed by source fit-style/contact just like LoBoFit.
    init_cp_np,init_n_np,_,_,_=nearest_surface(Vinit,target_tri,k=24)
    init_n=torch.tensor(init_n_np,dtype=dtype);init_v=torch.tensor(Vinit,dtype=dtype)

    cp=nrm=None
    hist=[]
    def decode():
        dabk=base_abk+delta
        nw=base_w+w_scale*torch.tanh(dw);nw=nw/(nw.sum(1,keepdim=True).clamp_min(1e-12))
        loc=dabk*tl[None,:,None]
        world=tj[None,:,:]+loc[:,:,0,None]*tx[None,:,:]+loc[:,:,1,None]*ty[None,:,:]+loc[:,:,2,None]*tz[None,:,:]
        return torch.sum(nw[:,:,None]*world,dim=1),nw

    for it in range(iterations+1):
        if cp is None or (it%contact_update==0):
            with torch.no_grad(): cur=decode()[0].detach().cpu().numpy()
            cp_np,n_np,_,_,_=nearest_surface(cur,target_tri,k=20);cp=torch.tensor(cp_np,dtype=dtype);nrm=torch.tensor(n_np,dtype=dtype)
        V,nw=decode()
        # Official-style preservation losses.
        lap=lap_delta_torch(V,E,cw);lap=lap/(torch.sqrt(torch.sum(lap*lap))+1e-12);Llap=torch.sum((lap-lap0)**2)
        a,b,c=V[Ft[:,0]],V[Ft[:,1]],V[Ft[:,2]];fn=torch.nn.functional.normalize(torch.linalg.cross(b-a,c-a),dim=1)
        if len(pairs_np):
            cos=torch.sum(fn[pairs[:,0]]*fn[pairs[:,1]],1);cos0=torch.sum(fn0[pairs[:,0]]*fn0[pairs[:,1]],1);Lb=torch.sum(pw*(cos-cos0)**2)
        else:Lb=torch.tensor(0.,dtype=dtype)
        if bt is not None:
            e1=V[bt[:,0]]-V[bt[:,1]];e2=V[bt[:,2]]-V[bt[:,1]];cur=torch.sum(torch.nn.functional.normalize(e1,dim=1)*torch.nn.functional.normalize(e2,dim=1),1);Lc=torch.sum(cwcur*(cur-curv0)**2)
        else:Lc=torch.tensor(0.,dtype=dtype)
        signed=torch.sum((V-cp)*nrm,dim=1)
        # separation is a soft barrier; source-fit tightness is the dominant target conform term
        sep=torch.relu(collision_margin-signed)
        Lsep=torch.sum(area_all*sep) if collision_area_weighted else torch.mean(sep)
        if len(fitids):Lfit=torch.sum(aw*(signed[fit_t]-ds_t)**2)
        else:Lfit=torch.tensor(0.,dtype=dtype)
        # Preserve semantic anatomical neighborhood without locking authored clearance: only
        # tangential drift from the body-correspondence initializer is penalized.
        dv=V-init_v; dn=torch.sum(dv*init_n,dim=1,keepdim=True);dtan=dv-dn*init_n
        Lsem=torch.mean(torch.sum(dtan*dtan,dim=1))/(semantic_tangent_scale**2)
        # Paper/official implementation specifically regularizes bone-axis (local z) residual and bounded weight residual.
        if official_reg_sums:
            Lk=torch.sum(delta[:,:,2]**2)*freeze_axis_strength
        else:
            Lk=torch.sum(delta[:,:,2]**2)/max(len(Vsrc),1)*freeze_axis_strength
        dwt=w_scale*torch.tanh(dw);Lw=torch.sum(dwt*dwt) if official_reg_sums else torch.sum(dwt*dwt)/max(len(Vsrc),1)
        # Match official ratios; fitness is rescaled from m^2 so it has useful strength at FFXIV scale.
        loss=collision_weight*Lsep + fit_weight*Lfit + bend_weight*Lb + curv_weight*Lc + lap_weight*Llap + 1.0*Lk + .01*Lw + semantic_tangent_weight*Lsem
        if it%log_every==0 or it==iterations:
            with torch.no_grad():
                q=V.detach().cpu().numpy();sd=signed.detach().cpu().numpy();
                hist.append({'iter':it,'loss':float(loss.detach()),'sep':float(Lsep.detach()),'fit':float(Lfit.detach()),'lap':float(Llap.detach()),'bend':float(Lb.detach()),'curv':float(Lc.detach()),'axis':float(Lk.detach()),'weight':float(Lw.detach()),'semantic_tangent':float(Lsem.detach()),'signed_med_mm':float(np.median(sd)*1000),'delta_rms':float(torch.sqrt(torch.mean(delta*delta)).detach()),'move_from_init_rms_mm':float(np.sqrt(np.mean(np.sum((q-Vinit)**2,axis=1)))*1000)})
        if it==iterations:break
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([delta,dw],10.0);opt.step()
    return decode()[0].detach().cpu().numpy(),hist,{'active_bones':[joint_names[i] for i in act],'fit_vertices':int(len(fitids))}
