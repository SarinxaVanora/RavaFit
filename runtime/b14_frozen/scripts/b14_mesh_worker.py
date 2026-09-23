from __future__ import annotations
import argparse,sys,json,time
from pathlib import Path
from collections import deque
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from ffxiv_lobofit import *
from structural_refine import *
from collision_eval import *
from lobofit_official_refine import refine_lobofit
from construction_fields import *

def orientation_stats(P,U,F):
    a0=np.cross(P[F[:,1]]-P[F[:,0]],P[F[:,2]]-P[F[:,0]]);a=np.cross(U[F[:,1]]-U[F[:,0]],U[F[:,2]]-U[F[:,0]])
    n0=np.linalg.norm(a0,axis=1);nn=np.linalg.norm(a,axis=1);dots=np.sum(a0*a,axis=1)/np.maximum(n0*nn,1e-12);ar=nn/np.maximum(n0,1e-12)
    return {'opposed':int(np.sum(dots<0)),'orient_min':float(np.min(dots)),'orient_p1':float(np.percentile(dots,1)),'area_min':float(np.min(ar)),'area_p1':float(np.percentile(ar,1))}

def solve_constructed_close(s,t,w,Wg,base,X,Y,BW,NS,NT,names,axes,tree,st,tt,lab,feat):
    Uconst=suppress_bone_axis_drift(w['V'],base,Wg,axes,.60);cid=feat['root_component'];ids=np.where(lab==cid)[0];F=clean_component_faces(w['F'],ids);P=w['V'][ids];WG=Wg[ids]
    dist,idx=tree.query(P,k=min(192,len(X)));dist=dist if dist.ndim>1 else dist[:,None];idx=idx if idx.ndim>1 else idx[:,None]
    align=np.einsum('nk,nqk->nq',WG,BW[idx]);sx=np.sign(P[:,0])[:,None];bx=np.sign(X[idx][:,:,0]);side=((np.abs(P[:,0,None])>.02)&(sx!=bx));sigma=.015
    ww=np.exp(-.5*(dist/sigma)**2)*np.power(.05+np.maximum(align,0),2.);ww=np.where(side,ww*1e-4,ww);ww/=np.maximum(ww.sum(1,keepdims=True),1e-12)
    local=P+np.sum((Y-X)[idx]*ww[:,:,None],axis=1)
    cs=P.mean(0);ct=local.mean(0);A0=P-cs;B0=local-ct;C=A0.T@A0;D=B0.T@A0;lam=.02*np.trace(C)/3;Af=(D+lam*np.eye(3))@np.linalg.inv(C+lam*np.eye(3));uu,sv,vt=np.linalg.svd(Af);Af=uu@np.diag(np.clip(sv,.25,1.5))@vt;global_map=(P-cs)@Af.T+ct
    J,_,_=skeleton_global_positions(s,names);up=J[names.index('j_kubi')]-J[names.index('j_kosi')];up/=np.linalg.norm(up)
    bg=body_guided_contacts(P,WG,X,Y,BW,NS,NT,k=24,bone_penalty=.012);body_change=np.linalg.norm(bg['target_point']-bg['source_point'],axis=1)
    du=np.sum(((global_map-P)-(local-P))*up[None,:],axis=1);alpha=np.exp(-np.square(body_change/.020));root0=local+(1.10*alpha*du)[:,None]*up[None,:]
    _,nr,_,_,_=nearest_surface(root0,tt,k=24);cfg={'lap':.35,'bend':.8,'curv':1.0,'shape':20.,'normal':.12,'reg':.15,'disp_smooth':7.,'plane':0}
    pre,h0=refine_component(P,root0,F,nr,iterations=280,lr=.016,tangent_cap=.005,normal_cap=.003,weights=cfg,log_every=280)
    pre,cage=boundary_similarity_cage(P,pre,F,strength=.30,decay_rings=3.0)
    loops=boundary_loops(F);E=unique_edges(F);nbr=[[] for _ in range(len(P))]
    for a,b in E:nbr[int(a)].append(int(b));nbr[int(b)].append(int(a))
    ring=np.full(len(P),999,dtype=int);owner=np.full(len(P),-1,dtype=int);q=deque()
    for li,L in enumerate(loops):
        for x in L:
            x=int(x)
            if ring[x]>0:ring[x]=0;owner[x]=li;q.append(x)
    while q:
        x=q.popleft()
        if ring[x]>=29:continue
        for nb in nbr[x]:
            if ring[nb]>ring[x]+1:ring[nb]=ring[x]+1;owner[nb]=owner[x];q.append(nb)
    fitmask=ring<=2;_,_,source_sd,_,_=nearest_surface(P,st,k=24);loop_change=np.array([np.median(body_change[L]) for L in loops]);loop_strength=1.-np.exp(-np.square(loop_change/.025));ab=np.abs(source_sd);xx=ab/.005;sat=np.ones_like(ab);nz=ab>1e-9;sat[nz]=np.tanh(xx[nz])/xx[nz];strength=np.where(owner>=0,loop_strength[np.maximum(owner,0)],0.);clear_scale=1.-strength*(1.-sat)
    R,h,meta=refine_lobofit(P,pre,F,WG,names,s,st,tt,iterations=460,lr=.0018,contact_update=75,log_every=460,fit_weight=3000.,semantic_tangent_weight=0.,collision_margin=.00035,lap_weight=.5,bend_weight=1.5,curv_weight=1.2,official_reg_sums=True,collision_area_weighted=True,fit_vertex_mask=fitmask,fit_clearance_scale=clear_scale)
    R,ci=smooth_collision_polish(R,F,tt,margin=.00035,iterations=10,blend=.45,max_push=.0015);U=Uconst.copy();U[ids]=R
    U,kinds,edges,ah=refine_assembly(w['V'],U,w['F'],lab,root_component=cid,iterations=420,lr=.018,max_gap=.003,log_every=420)
    stage={'mode':'constructed_close_shell','boundary_similarity_cage':cage,'longitudinal_regrade':1.10,'fit_boundary_rings':2,'pre_last':h0[-1],'lobofit_last':h[-1],'lobofit_meta':meta,'collision':ci,'assembly_last':ah[-1]}
    return U,ids,F,stage

def solve_standoff(s,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,st,tt,lab,feat):
    cid=feat['root_component'];ids=np.where(lab==cid)[0];F=clean_component_faces(w['F'],ids);P=w['V'][ids];WG=Wg[ids]
    U=suppress_bone_axis_drift(w['V'],base,Wg,axes,.65)
    Uroot,di=diffuse_body_tangent(P,U[ids],F,NS[contact[ids]],strength=.28,iterations=20,blend=.5)
    Uroot,cage=boundary_similarity_cage(P,Uroot,F,strength=.50,decay_rings=3.0);U[ids]=Uroot
    Pref,re=longitudinal_loop_regrade(P,F,WG,X,Y,BW,NS,NT,s,names,scale=.65,statistic='median')
    _,nr,_,_,_=nearest_surface(U[ids],tt,k=24);cfg={'lap':8,'bend':1.5,'curv':1.5,'shape':2,'normal':2,'reg':.04,'disp_smooth':1.2,'plane':0}
    R,h=refine_component(Pref,U[ids],F,nr,iterations=500,lr=.025,tangent_cap=.026,weights=cfg,log_every=500,triangle_guard_reference=P,triangle_area_floor=.075,triangle_orient_floor=.075,triangle_guard_weight=1.0,triangle_guard_start=480);U[ids]=R
    R,ti=tightness_polish(P,U[ids],F,st,tt,strength=.20,max_adjust=.008);U[ids]=R;R,ci=smooth_collision_polish(U[ids],F,tt,margin=.00035,iterations=12,blend=.45,max_push=.0045);U[ids]=R
    U,kinds,edges,ah=refine_assembly(w['V'],U,w['F'],lab,root_component=cid,iterations=450,lr=.018,max_gap=.004,log_every=450)
    stage={'mode':'stand_off_structured_shell','tangent_diffusion':di,'boundary_similarity_cage':cage,'loop_regrade_scale':.65,'loop_motions_mm':(re['loop_motion']*1000).tolist(),'triangle_guard':{'area_floor':.075,'orientation_floor':.075,'weight':1.0,'start':480},'structural_last':h[-1],'tightness':ti,'collision':ci,'assembly_last':ah[-1]}
    return U,ids,F,stage

def solve_flexible(w,Wg,base,axes,tt,lab,feat):
    U=suppress_bone_axis_drift(w['V'],base,Wg,axes,.90);cid=feat['root_component'];ids=np.where(lab==cid)[0];F=clean_component_faces(w['F'],ids);P=w['V'][ids]
    _,nr,_,_,_=nearest_surface(U[ids],tt,k=24);cfg={'lap':8,'bend':1.5,'curv':1.5,'shape':2,'normal':2,'reg':.04,'disp_smooth':.8,'plane':0}
    R,h=refine_component(P,U[ids],F,nr,iterations=350,lr=.03,tangent_cap=.030,weights=cfg,log_every=350);U[ids]=R;R,ci=smooth_collision_polish(U[ids],F,tt,margin=.00035,iterations=18,blend=.55,max_push=.0025);U[ids]=R
    return U,ids,F,{'mode':'body_following_flexible_layer','axial_strength':.90,'structural_last':h[-1],'collision':ci}

def solve_conservative(w,Wg,base,lab,classes,feat):
    U=base.copy()
    for c,kind in classes.items():
        ids=np.where(lab==c)[0]
        if kind=='rigid' and len(ids)>=3:
            R,cs,cd=rigid_fit(w['V'][ids],base[ids],np.max(Wg[ids],axis=1)+.1);U[ids]=(w['V'][ids]-cs)@R.T+cd
    cid=feat['root_component'];ids=np.where(lab==cid)[0];F=clean_component_faces(w['F'],ids)
    return U,ids,F,{'mode':'conservative_component_assembly'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mesh');args=ap.parse_args();name=args.mesh;t0=time.time()
    s=GLB(ROOT/'inputs/Original_outfit.glb');t=GLB(ROOT/'inputs/Selected_Body.glb');target_before=sha(ROOT/'inputs/Selected_Body.glb')
    z=np.load(ROOT/'workers/body_cache.npz',allow_pickle=True);X=z['X'];Y=z['Y'];BW=z['BW'];NS=z['NS'];NT=z['NT'];A=z['A'];axes=z['axes'];names=z['names'].tolist();tree=cKDTree(X);st=body_triangles(s);tt=body_triangles(t)
    w=weld_mesh(s.data(name));Wg,_=solver_body_weights(w['V'],X,BW,tree);base,contact,*_=local_body_field_map_soft(w['V'],Wg,X,Y,BW,A,tree=tree,tau=.004);behavior,feat,lab,classes,details=infer_shell_behavior(w,st)
    if behavior=='stand_off_structured_shell':U,ids,F,stage=solve_standoff(s,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,st,tt,lab,feat)
    elif behavior=='constructed_close_shell':U,ids,F,stage=solve_constructed_close(s,t,w,Wg,base,X,Y,BW,NS,NT,names,axes,tree,st,tt,lab,feat)
    elif behavior=='body_following_flexible_layer':U,ids,F,stage=solve_flexible(w,Wg,base,axes,tt,lab,feat)
    else:U,ids,F,stage=solve_conservative(w,Wg,base,lab,classes,feat)
    raw=expand_welded(w,U); outnpz=ROOT/'workers'/f"{name.replace(' ','_').replace('.','_')}_B14.npz";np.savez_compressed(outnpz,raw=raw,weld=U,root_ids=ids)
    # Optional validation is post-freeze only and is never required by the production solve.
    valpath=ROOT/'diagnostics/Validation_proxy_on_Selected.glb'
    if valpath.exists():
        val=GLB(valpath);met=chamfer(raw,val.data(name)['V']) if name in val.mesh_names() else None
    else:met=None
    orient=orientation_stats(w['V'][ids],U[ids],F) if len(F) else {};qr=quality(w['V'][ids],U[ids],F) if len(F) else {}
    record={'mesh':name,'fresh_from_original':True,'generation_uses_validation':False,'behavior':behavior,'features':feat,'metrics':met,'quality':qr,'orientation':orient,'stage':stage,'elapsed_sec':time.time()-t0,'source_hash':sha(ROOT/'inputs/Original_outfit.glb'),'target_hash_before':target_before,'target_hash_after':sha(ROOT/'inputs/Selected_Body.glb'),'target_unchanged':target_before==sha(ROOT/'inputs/Selected_Body.glb')}
    jout=ROOT/'workers'/f"{name.replace(' ','_').replace('.','_')}_B14.json";jout.write_text(json.dumps(record,indent=2));print(json.dumps({'mesh':name,'behavior':behavior,'metrics':met,'orientation':orient,'elapsed_sec':record['elapsed_sec']}))
if __name__=='__main__':main()
