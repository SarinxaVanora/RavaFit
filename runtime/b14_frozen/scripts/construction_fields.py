from __future__ import annotations
import numpy as np
from collections import defaultdict, deque
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from structural_refine import unique_edges, clean_component_faces
from collision_eval import nearest_surface


def boundary_loops(F):
    F=np.asarray(F,np.int64)
    all_e=np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1)
    ue,cnt=np.unique(all_e,axis=0,return_counts=True); be=ue[cnt==1]
    adj=defaultdict(list)
    for a,b in be: adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    loops=[]; seen=set()
    for st in adj:
        if st in seen: continue
        stack=[st];seen.add(st);comp=[]
        while stack:
            q=stack.pop();comp.append(q)
            for v in adj[q]:
                if v not in seen: seen.add(v);stack.append(v)
        loops.append(np.asarray(comp,dtype=np.int64))
    return loops


def harmonic_boundary_field(n,F,loops,values):
    E=unique_edges(F); rows=[];cols=[];data=[]
    for a,b in E:
        a=int(a);b=int(b);rows += [a,a,b,b]; cols += [a,b,b,a]; data += [1.,-1.,1.,-1.]
    L=coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr()
    boundary=np.zeros(n,dtype=bool); bval=np.zeros(n,float)
    for i,loop in enumerate(loops): boundary[loop]=True; bval[loop]=values[i]
    interior=np.where(~boundary)[0]; bidx=np.where(boundary)[0]; field=bval.copy()
    if len(interior) and len(bidx): field[interior]=spsolve(L[interior][:,interior],-L[interior][:,bidx]@bval[bidx])
    lim=max(float(np.percentile(np.abs(values),95)),1e-6); return np.clip(field,-lim,lim)


def longitudinal_loop_regrade(P,F,Wg,X,Y,BW,NS,NT,rig_glb,joint_names,scale=.65,statistic='median'):
    from ffxiv_lobofit import body_guided_contacts, skeleton_global_positions
    J,_,_=skeleton_global_positions(rig_glb,joint_names)
    up=J[joint_names.index('j_kubi')]-J[joint_names.index('j_kosi')]; up/=max(np.linalg.norm(up),1e-12)
    bg=body_guided_contacts(P,Wg,X,Y,BW,NS,NT,k=24,bone_penalty=.012)
    body_du=(bg['target_point']-bg['source_point'])@up
    loops=boundary_loops(F)
    if statistic=='mean': vals=np.array([np.mean(body_du[L]) for L in loops],float)
    else: vals=np.array([np.median(body_du[L]) for L in loops],float)
    field=harmonic_boundary_field(len(P),F,loops,vals)
    return P+(scale*field)[:,None]*up[None,:], {'axis':up,'loops':loops,'loop_motion':vals,'field':field,'body_du':body_du}


def diffuse_body_tangent(P,U,F,source_normals,strength=.28,iterations=20,blend=.5):
    P=np.asarray(P,float); U=np.asarray(U,float); N=np.asarray(source_normals,float).copy(); N/=np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)
    D=U-P; dn=np.sum(D*N,axis=1,keepdims=True)*N; dt=D-dn
    E=unique_edges(F); nbr=[[] for _ in range(len(P))]
    for a,b in E: nbr[int(a)].append(int(b)); nbr[int(b)].append(int(a))
    sm=dt.copy()
    for _ in range(iterations):
        avg=np.array([np.mean(sm[nb],axis=0) if nb else sm[i] for i,nb in enumerate(nbr)])
        sm=(1-blend)*sm+blend*avg
        sm-=np.sum(sm*N,axis=1,keepdims=True)*N
    dtf=(1-strength)*dt+strength*sm
    return P+dn+dtf, {'strength':strength,'iterations':iterations,'blend':blend,'tangent_rms_before_mm':float(np.sqrt(np.mean(np.sum(dt*dt,axis=1)))*1000),'tangent_rms_after_mm':float(np.sqrt(np.mean(np.sum(dtf*dtf,axis=1)))*1000)}


def infer_shell_behavior(w,source_tri):
    from ffxiv_lobofit import classify_components
    lab,classes,details=classify_components(w); cid=int(np.argmax(np.bincount(lab))); ids=np.where(lab==cid)[0]
    F=clean_component_faces(w['F'],ids); P=w['V'][ids]
    _,_,ds,_,_=nearest_surface(P,source_tri,k=24)
    loops=boundary_loops(F); bverts=np.unique(np.concatenate(loops)) if loops else np.array([],dtype=int)
    root_fraction=float(len(ids)/max(len(w['V']),1)); boundary_fraction=float(len(bverts)/max(len(P),1)); clearance_median=float(np.median(np.abs(ds)))
    # Source-only routing. Thresholds represent geometry/contact regimes, not garment identities.
    if root_fraction>=0.80 and boundary_fraction<=0.08:
        behavior='body_following_flexible_layer'
    elif boundary_fraction>=0.10 and clearance_median>=0.006:
        behavior='stand_off_structured_shell'
    elif boundary_fraction>=0.10 and clearance_median<0.006 and root_fraction>=0.30:
        behavior='constructed_close_shell'
    else:
        behavior='conservative_component_assembly'
    return behavior, {'root_component':cid,'root_vertices':int(len(ids)),'root_fraction':root_fraction,'boundary_fraction':boundary_fraction,'source_clearance_median_mm':clearance_median*1000,'component_count':int(lab.max()+1),'boundary_loops':int(len(loops))}, lab, classes, details


def similarity_transform(src,dst,scale_min=.6,scale_max=1.4):
    src=np.asarray(src,float);dst=np.asarray(dst,float);cs=src.mean(0);cd=dst.mean(0);A=src-cs;B=dst-cd;H=A.T@B;u,sv,vt=np.linalg.svd(H);R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    Ar=A@R.T;scale=float(np.sum(Ar*B)/max(np.sum(Ar*Ar),1e-12));scale=float(np.clip(scale,scale_min,scale_max));return scale,R,cs,cd


def boundary_similarity_cage(P,U,F,strength=.5,decay_rings=3.0):
    """Restore authored open-boundary shape up to a source->current similarity transform.

    The body-driven U determines each loop's new placement/rotation/scale. Only local loop
    distortion is removed; its correction then decays smoothly into the panel interior.
    """
    P=np.asarray(P,float);U=np.asarray(U,float);loops=boundary_loops(F);n=len(P);bmask=np.zeros(n,bool);C=np.zeros_like(P);info=[]
    for L in loops:
        sc,R,cs,cd=similarity_transform(P[L],U[L]);coh=sc*(P[L]-cs)@R.T+cd;corr=coh-U[L];C[L]=corr;bmask[L]=True;info.append({'n':int(len(L)),'scale':sc,'correction_rms_mm':float(np.sqrt(np.mean(np.sum(corr*corr,axis=1)))*1000)})
    if not np.any(bmask):return U.copy(),{'strength':strength,'decay_rings':decay_rings,'loops':info}
    E=unique_edges(F);rows=[];cols=[];data=[]
    for a,b in E:
        a=int(a);b=int(b);rows += [a,a,b,b];cols += [a,b,b,a];data += [1.,-1.,1.,-1.]
    Lm=coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr();ii=np.where(~bmask)[0];bb=np.where(bmask)[0]
    if len(ii):
        for k in range(3):C[ii,k]=spsolve(Lm[ii][:,ii],-Lm[ii][:,bb]@C[bb,k])
    nbr=[[] for _ in range(n)]
    for a,b in E:nbr[int(a)].append(int(b));nbr[int(b)].append(int(a))
    ring=np.full(n,999,dtype=int);q=deque()
    for i in bb:ring[i]=0;q.append(int(i))
    while q:
        x=q.popleft()
        for nb in nbr[x]:
            if ring[nb]>ring[x]+1:ring[nb]=ring[x]+1;q.append(nb)
    decay=np.exp(-ring/max(float(decay_rings),1e-6));out=U+float(strength)*decay[:,None]*C
    return out,{'strength':float(strength),'decay_rings':float(decay_rings),'loops':info,'correction_rms_mm':float(np.sqrt(np.mean(np.sum((out-U)**2,axis=1)))*1000)}
