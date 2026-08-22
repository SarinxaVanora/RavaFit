import numpy as np, trimesh
from scipy.spatial import cKDTree

def body_triangles(glb):
    tris=[]
    for name in glb.mesh_names():
        if not name or not name.startswith('mesh 0.'): continue
        d=glb.data(name)
        if 'bibo' not in d['material'].lower(): continue
        tris.append(d['V'][d['F']])
    return np.concatenate(tris,axis=0)

def nearest_surface(P,triangles,k=16):
    P=np.asarray(P,float);T=np.asarray(triangles,float);cent=T.mean(1);tree=cKDTree(cent);_,idx=tree.query(P,k=min(k,len(T)));idx=idx if idx.ndim>1 else idx[:,None]
    n=len(P);kk=idx.shape[1];Tc=T[idx.reshape(-1)];Q=np.repeat(P,kk,axis=0);C=trimesh.triangles.closest_point(Tc,Q).reshape(n,kk,3);dd=np.sum((C-P[:,None,:])**2,axis=2);j=np.argmin(dd,axis=1);cp=C[np.arange(n),j];fi=idx[np.arange(n),j]
    fn=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);fn/=np.maximum(np.linalg.norm(fn,axis=1,keepdims=True),1e-12);N=fn[fi]
    signed=np.sum((P-cp)*N,axis=1);dist=np.sqrt(np.min(dd,axis=1));return cp,N,signed,dist,fi

def evaluate(P,T,k=16,epsilon=.0003):
    cp,n,s,d,fi=nearest_surface(P,T,k);return {'signed':s,'distance':d,'penetration_fraction':float(np.mean(s < -epsilon)),'penetration_p01_mm':float(np.percentile(s,1)*1000),'signed_median_mm':float(np.median(s)*1000),'distance_median_mm':float(np.median(d)*1000),'distance_p95_mm':float(np.percentile(d,95)*1000)}

def smooth_collision_polish(V,F,triangles,margin=.00035,iterations=18,blend=.55,max_push=.0025,k=16):
    import numpy as np
    cp,n,s,d,fi=nearest_surface(V,triangles,k=k)
    req=np.maximum(0.0,margin-s)
    # adjacency on physical topology
    E=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
    nbr=[[] for _ in range(len(V))]
    for a,b in E:nbr[a].append(b);nbr[b].append(a)
    push=req.copy()
    for _ in range(iterations):
        avg=np.array([np.mean(push[x]) if x else push[i] for i,x in enumerate(nbr)])
        # penetration requirement is a lower bound; surrounding vertices receive a much weaker smooth carry.
        push=np.maximum(req, blend*push+(1-blend)*avg*.55)
    push=np.clip(push,0,max_push)
    out=V+n*push[:,None]
    return out,{'before_pen_frac':float(np.mean(s < -0.0003)),'before_min_mm':float(np.min(s)*1000),'push_rms_mm':float(np.sqrt(np.mean(push*push))*1000),'push_max_mm':float(np.max(push)*1000),'affected':int(np.sum(push>1e-6))}

def smooth_scalar_field(values,F,retain=.72,iterations=12):
    values=np.asarray(values,float);E=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0);nbr=[[] for _ in range(len(values))]
    for a,b in E:nbr[a].append(b);nbr[b].append(a)
    cur=values.copy()
    for _ in range(iterations):
        avg=np.array([np.mean(cur[x]) if x else cur[i] for i,x in enumerate(nbr)]);cur=retain*values+(1-retain)*avg
    return cur

def tightness_polish(Vsrc,V,F,source_tri,target_tri,strength=.75,max_adjust=.008,k=16,clearance_scale=None):
    cps,ns,ds,es,fis=nearest_surface(Vsrc,source_tri,k=k);cpt,nt,dt,et,fit=nearest_surface(V,target_tri,k=k)
    if clearance_scale is None:
        desired_ds=ds
    else:
        sc=np.asarray(clearance_scale,float)
        if sc.shape!=(len(Vsrc),): raise ValueError(f'clearance_scale shape {sc.shape} != {(len(Vsrc),)}')
        desired_ds=ds*sc
    desired=np.clip(desired_ds-dt,-max_adjust,max_adjust);field=smooth_scalar_field(desired,F,retain=.72,iterations=12)*strength
    out=V+nt*field[:,None]
    return out,{'source_signed_median_mm':float(np.median(ds)*1000),'desired_signed_median_mm':float(np.median(desired_ds)*1000),'before_signed_median_mm':float(np.median(dt)*1000),'desired_median_mm':float(np.median(desired)*1000),'applied_rms_mm':float(np.sqrt(np.mean(field*field))*1000),'applied_p95_mm':float(np.percentile(np.abs(field),95)*1000)}
