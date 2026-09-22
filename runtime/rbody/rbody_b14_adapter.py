from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def _unit(v):
    x=np.asarray(v,dtype=np.float64);n=np.linalg.norm(x,axis=1,keepdims=True);return x/np.maximum(n,1e-12)


def _hybrid_correspondence(source_ref,target_ref,k=32):
    suv=np.asarray(source_ref['UV'],dtype=np.float64);tuv=np.asarray(target_ref['UV'],dtype=np.float64)
    sv=np.asarray(source_ref['V'],dtype=np.float64);tv=np.asarray(target_ref['V'],dtype=np.float64)
    sw=np.asarray(source_ref['W'],dtype=np.float64);tw=np.asarray(target_ref['W'],dtype=np.float64);tn=_unit(target_ref['N'])
    if len(suv)==0 or len(tuv)==0:raise ValueError('Source/target body reference has no UV data')
    if sw.ndim!=2 or tw.ndim!=2 or sw.shape[1]!=tw.shape[1]:raise ValueError('Source/target body weights are not in the same rig space')

    shared=np.sum(tw,axis=1)>1e-12
    shared_ids=np.flatnonzero(shared)
    if len(shared_ids)<8:raise ValueError('Target body has too little shared rig-weight surface for correspondence')
    kk=min(max(int(k),8),len(shared_ids));uv_tree=cKDTree(tuv[shared_ids]);spatial_tree=cKDTree(tv[shared_ids])
    uv_dist,uv_local=uv_tree.query(suv,k=kk);spatial_dist,spatial_local=spatial_tree.query(sv,k=kk)
    uv_idx=shared_ids[np.asarray(uv_local,dtype=np.int64)];spatial_idx=shared_ids[np.asarray(spatial_local,dtype=np.int64)]
    if uv_dist.ndim==1:uv_dist=uv_dist[:,None];uv_idx=uv_idx[:,None]
    if spatial_dist.ndim==1:spatial_dist=spatial_dist[:,None];spatial_idx=spatial_idx[:,None]

    nearest_spatial=np.asarray(spatial_dist[:,0],dtype=np.float64);spatial_scale=float(np.clip(max(0.020,np.percentile(nearest_spatial,90)*2.0),0.020,0.060))
    uv_scale=0.010;bone_weight=2.5;output=np.zeros_like(sv,dtype=np.float64);normals=np.zeros_like(sv,dtype=np.float64)
    chosen_spatial=np.zeros(len(sv),dtype=np.float64);chosen_alignment=np.zeros(len(sv),dtype=np.float64)
    for row in range(len(sv)):
        candidates=np.unique(np.concatenate((uv_idx[row],spatial_idx[row])));duv=np.linalg.norm(tuv[candidates]-suv[row],axis=1);dsp=np.linalg.norm(tv[candidates]-sv[row],axis=1);align=tw[candidates]@sw[row]
        uv_authority=1.0 if float(uv_dist[row,0])<0.025 else 0.15;score=(uv_authority*np.minimum(duv/uv_scale,4.0))+(dsp/spatial_scale)+(bone_weight*(1.0-align))
        order=np.argsort(score)[:min(6,len(score))];chosen=candidates[order];local_score=score[order];blend=np.exp(-(local_score-local_score.min())*2.0);blend/=np.maximum(blend.sum(),1e-12)
        output[row]=(tv[chosen]*blend[:,None]).sum(0);normal=(tn[chosen]*blend[:,None]).sum(0);normal/=max(float(np.linalg.norm(normal)),1e-12);normals[row]=normal
        chosen_spatial[row]=float(np.min(dsp[order]));chosen_alignment[row]=float(np.max(align[order]))
    return output,normals,{'uv_nearest':np.asarray(uv_dist[:,0],dtype=np.float64),'chosen_spatial':chosen_spatial,'chosen_alignment':chosen_alignment,'spatial_scale':spatial_scale}


def _spatial_rig_correspondence(source_ref,target_ref,k=48):
    sv=np.asarray(source_ref['V'],dtype=np.float64);tv=np.asarray(target_ref['V'],dtype=np.float64)
    sw=np.asarray(source_ref['W'],dtype=np.float64);tw=np.asarray(target_ref['W'],dtype=np.float64);tn=_unit(target_ref['N'])
    if sw.ndim!=2 or tw.ndim!=2 or sw.shape[1]!=tw.shape[1]:raise ValueError('Source/target body weights are not in the same rig space')
    shared=np.sum(tw,axis=1)>1e-12
    shared_ids=np.flatnonzero(shared)
    if len(shared_ids)<8:raise ValueError('Target body has too little shared rig-weight surface for spatial correspondence')
    kk=min(max(int(k),8),len(shared_ids));tree=cKDTree(tv[shared_ids]);spatial_dist,spatial_local=tree.query(sv,k=kk);spatial_idx=shared_ids[np.asarray(spatial_local,dtype=np.int64)]
    if spatial_dist.ndim==1:spatial_dist=spatial_dist[:,None];spatial_idx=spatial_idx[:,None]
    nearest=np.asarray(spatial_dist[:,0],dtype=np.float64);spatial_scale=float(np.clip(max(0.020,np.percentile(nearest,90)*2.0),0.020,0.060))
    output=np.zeros_like(sv,dtype=np.float64);normals=np.zeros_like(sv,dtype=np.float64);chosen_alignment=np.zeros(len(sv),dtype=np.float64);chosen_spatial=np.zeros(len(sv),dtype=np.float64)
    for first in range(0,len(sv),1024):
        last=min(first+1024,len(sv));candidates=spatial_idx[first:last];dsp=spatial_dist[first:last];source_weights=sw[first:last]
        candidate_weights=tw[candidates];align=np.einsum('bkj,bj->bk',candidate_weights,source_weights,optimize=True);score=(dsp/spatial_scale)+(3.0*(1.0-align))
        sx=sv[first:last,0];crossed=(np.sign(tv[candidates,0])!=np.sign(sx[:,None]))&(np.abs(tv[candidates,0])>0.005)&(np.abs(sx[:,None])>0.025)
        score+=crossed.astype(np.float64)*1.5*np.minimum(np.abs(sx[:,None])/0.05,3.0)
        take=min(6,score.shape[1]);order=np.argpartition(score,take-1,axis=1)[:,:take];local_score=np.take_along_axis(score,order,axis=1);chosen=np.take_along_axis(candidates,order,axis=1);local_dsp=np.take_along_axis(dsp,order,axis=1);local_align=np.take_along_axis(align,order,axis=1)
        base=np.min(local_score,axis=1,keepdims=True);blend=np.exp(-(local_score-base)*2.0);blend/=np.maximum(np.sum(blend,axis=1,keepdims=True),1e-12)
        output[first:last]=np.sum(tv[chosen]*blend[:,:,None],axis=1);normal=np.sum(tn[chosen]*blend[:,:,None],axis=1);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12);normals[first:last]=normal
        chosen_spatial[first:last]=np.min(local_dsp,axis=1);chosen_alignment[first:last]=np.max(local_align,axis=1)
    return output,normals,{'chosen_spatial':chosen_spatial,'chosen_alignment':chosen_alignment,'spatial_scale':spatial_scale}


def _cross_sex_rig_correspondence(source_ref,target_ref,k=64):
    """Cross-sex Legs correspondence using a body-derived shared-rig semantic alignment.

    The male side of the pair is expected to use a smallclothes support surface.  Source and
    target meshes can have unrelated topology/UVs, so candidate search is performed after a
    per-joint weighted-centroid alignment, then constrained by the existing rig-weight signature.
    This is intentionally a pair-building concern: garment structure is still solved by B14.
    """
    sv=np.asarray(source_ref['V'],dtype=np.float64);tv=np.asarray(target_ref['V'],dtype=np.float64)
    sw=np.asarray(source_ref['W'],dtype=np.float64);tw=np.asarray(target_ref['W'],dtype=np.float64);tn=_unit(target_ref['N'])
    if sw.ndim!=2 or tw.ndim!=2 or sw.shape[1]!=tw.shape[1]:raise ValueError('Cross-sex source/target body weights are not in the same rig space')
    if len(sv)<8 or len(tv)<8:raise ValueError('Cross-sex correspondence requires non-empty source and target support surfaces')

    joint_delta=np.zeros((sw.shape[1],3),dtype=np.float64);valid=np.zeros(sw.shape[1],dtype=bool)
    for column in range(sw.shape[1]):
        source_mass=float(np.sum(sw[:,column]));target_mass=float(np.sum(tw[:,column]))
        if source_mass<=1e-5 or target_mass<=1e-5:continue
        source_centre=np.sum(sv*sw[:,column,None],axis=0)/source_mass;target_centre=np.sum(tv*tw[:,column,None],axis=0)/target_mass
        delta=target_centre-source_centre
        # Weighted centroids are only a semantic pre-alignment.  Prevent a pathological remote
        # auxiliary joint from dragging a whole region before the actual body correspondence.
        length=float(np.linalg.norm(delta))
        if length>.120:delta*=.120/max(length,1e-12)
        joint_delta[column]=delta;valid[column]=True
    if np.count_nonzero(valid)<3:
        return _spatial_rig_correspondence(source_ref,target_ref,k=max(48,int(k)))

    aligned=sv+sw[:,valid]@joint_delta[valid]
    shared=np.sum(tw,axis=1)>1e-12;shared_ids=np.flatnonzero(shared)
    kk=min(max(int(k),16),len(shared_ids));tree=cKDTree(tv[shared_ids]);distance,local=tree.query(aligned,k=kk);candidates=shared_ids[np.asarray(local,dtype=np.int64)]
    if distance.ndim==1:distance=distance[:,None];candidates=candidates[:,None]
    candidate_weights=tw[candidates];alignment=np.einsum('nkj,nj->nk',candidate_weights,sw,optimize=True)
    spatial_scale=float(np.clip(max(.012,np.percentile(distance[:,0],90)*1.75),.012,.055))
    score=(distance/max(spatial_scale,1e-9))+(3.25*(1.0-alignment))
    sx=aligned[:,0];crossed=(np.sign(tv[candidates,0])!=np.sign(sx[:,None]))&(np.abs(tv[candidates,0])>.005)&(np.abs(sx[:,None])>.020)
    score+=crossed.astype(np.float64)*2.5
    take=min(8,score.shape[1]);order=np.argpartition(score,take-1,axis=1)[:,:take];local_score=np.take_along_axis(score,order,axis=1);chosen=np.take_along_axis(candidates,order,axis=1)
    local_distance=np.take_along_axis(distance,order,axis=1);local_alignment=np.take_along_axis(alignment,order,axis=1)
    base=np.min(local_score,axis=1,keepdims=True);blend=np.exp(-(local_score-base)*2.25);blend/=np.maximum(np.sum(blend,axis=1,keepdims=True),1e-12)
    output=np.sum(tv[chosen]*blend[:,:,None],axis=1);normals=np.sum(tn[chosen]*blend[:,:,None],axis=1);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    semantic_shift=np.linalg.norm(aligned-sv,axis=1)
    return output,normals,{
        'chosen_spatial':np.min(local_distance,axis=1),'chosen_alignment':np.max(local_alignment,axis=1),'spatial_scale':spatial_scale,
        'mode':'cross_sex_rig_smallclothes','uv_nearest':np.full(len(sv),np.nan,dtype=np.float64),
        'semantic_joint_count':int(np.count_nonzero(valid)),'semantic_shift_p50_mm':float(np.percentile(semantic_shift,50)*1000.0),'semantic_shift_p95_mm':float(np.percentile(semantic_shift,95)*1000.0),'semantic_shift_max_mm':float(np.max(semantic_shift)*1000.0),
        'hybrid_quality':{},'spatial_quality':{},
    }


def _correspondence_quality(source_ref,mapped):
    X=np.asarray(source_ref['V'],dtype=np.float64);F=np.asarray(source_ref['F'],dtype=np.int64);Y=np.asarray(mapped,dtype=np.float64)
    quality=_surface_quality(X,Y,F);quality['displacement_rms_mm']=float(np.sqrt(np.mean(np.sum((Y-X)**2,axis=1)))*1000.0);return quality


def _prefer_spatial_correspondence(hybrid,spatial):
    hybrid_bad=(hybrid['orientation_flip_fraction']>=0.060) or (hybrid['edge_stretch_over_3x_fraction']>=0.020)
    topology_better=(spatial['orientation_flip_fraction']<=hybrid['orientation_flip_fraction']*0.70) and (spatial['edge_stretch_over_3x_fraction']<=hybrid['edge_stretch_over_3x_fraction']*0.70)
    displacement_sane=spatial['displacement_rms_mm']<=hybrid['displacement_rms_mm']*1.10
    return bool(hybrid_bad and topology_better and displacement_sane)


def uv_correspondence(source_ref,target_ref,k=32):
    """Map a source body to its target, rejecting pathological UV correspondence when necessary."""
    hybrid_Y,hybrid_N,hybrid_diag=_hybrid_correspondence(source_ref,target_ref,k=k);hybrid_quality=_correspondence_quality(source_ref,hybrid_Y)
    hybrid_bad=(hybrid_quality['orientation_flip_fraction']>=0.060) or (hybrid_quality['edge_stretch_over_3x_fraction']>=0.020)
    spatial_quality={}
    if hybrid_bad:
        spatial_Y,spatial_N,spatial_diag=_spatial_rig_correspondence(source_ref,target_ref,k=max(48,int(k)));spatial_quality=_correspondence_quality(source_ref,spatial_Y);use_spatial=_prefer_spatial_correspondence(hybrid_quality,spatial_quality)
    else:
        use_spatial=False
    if use_spatial:
        output,normals=spatial_Y,spatial_N;chosen_spatial=spatial_diag['chosen_spatial'];chosen_alignment=spatial_diag['chosen_alignment'];spatial_scale=spatial_diag['spatial_scale'];mode='spatial_rig'
    else:
        output,normals=hybrid_Y,hybrid_N;chosen_spatial=hybrid_diag['chosen_spatial'];chosen_alignment=hybrid_diag['chosen_alignment'];spatial_scale=hybrid_diag['spatial_scale'];mode='uv_rig_spatial'
    diagnostics={'uv_nearest':hybrid_diag['uv_nearest'],'chosen_spatial':chosen_spatial,'chosen_alignment':chosen_alignment,'spatial_scale':spatial_scale,'mode':mode,'hybrid_quality':hybrid_quality,'spatial_quality':spatial_quality}
    return output,normals,diagnostics


def _weld_support(V,F,Y,W,tolerance=2e-5):
    V=np.asarray(V,dtype=np.float64);F=np.asarray(F,dtype=np.int64);Y=np.asarray(Y,dtype=np.float64);W=np.asarray(W,dtype=np.float64)
    keys=np.round(V/float(tolerance)).astype(np.int64);index={};inverse=np.empty(len(V),dtype=np.int64);groups=[]
    for row,key in enumerate(map(tuple,keys)):
        welded=index.get(key)
        if welded is None:
            welded=len(groups);index[key]=welded;groups.append([])
        inverse[row]=welded;groups[welded].append(row)
    count=len(groups);WV=np.zeros((count,3),dtype=np.float64);WY=np.zeros((count,3),dtype=np.float64);WW=np.zeros((count,W.shape[1]),dtype=np.float64)
    for welded,rows in enumerate(groups):
        ids=np.asarray(rows,dtype=np.int64);WV[welded]=np.mean(V[ids],axis=0);WY[welded]=np.mean(Y[ids],axis=0);WW[welded]=np.mean(W[ids],axis=0)
        total=float(np.sum(WW[welded]));
        if total>1e-12:WW[welded]/=total
    WF=inverse[F];good=(WF[:,0]!=WF[:,1])&(WF[:,1]!=WF[:,2])&(WF[:,2]!=WF[:,0]);WF=WF[good]
    return WV,WF,WY,WW,inverse


def _vertex_normals(V,F):
    V=np.asarray(V,dtype=np.float64);F=np.asarray(F,dtype=np.int64);out=np.zeros_like(V)
    if len(F):
        normals=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
        np.add.at(out,F[:,0],normals);np.add.at(out,F[:,1],normals);np.add.at(out,F[:,2],normals)
    length=np.linalg.norm(out,axis=1,keepdims=True);out/=np.maximum(length,1e-12);return out


def _boundary_groups(V,F):
    directed={};counts={}
    for tri in np.asarray(F,dtype=np.int64):
        for a,b in ((int(tri[0]),int(tri[1])),(int(tri[1]),int(tri[2])),(int(tri[2]),int(tri[0]))):
            key=(a,b) if a<b else (b,a);counts[key]=counts.get(key,0)+1;directed.setdefault(key,(a,b))
    boundary=[directed[key] for key,count in counts.items() if count==1]
    adjacency={}
    for a,b in boundary:adjacency.setdefault(a,set()).add(b);adjacency.setdefault(b,set()).add(a)
    seen=set();groups=[]
    for first in adjacency:
        if first in seen:continue
        stack=[first];seen.add(first);vertices=[]
        while stack:
            current=stack.pop();vertices.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in seen:seen.add(neighbour);stack.append(neighbour)
        vertex_set=set(vertices);edges=[edge for edge in boundary if edge[0] in vertex_set and edge[1] in vertex_set]
        groups.append((np.asarray(vertices,dtype=np.int64),edges))
    return groups


def _outer_boundary(V,ids):
    points=V[ids];lo=np.min(V,axis=0);hi=np.max(V,axis=0);span=np.maximum(hi-lo,1e-6)
    x_margin=max(0.006,float(span[0])*0.12);y_margin=max(0.006,float(span[1])*0.12)
    near_x=(points[:,0]-lo[0]<=x_margin)|(hi[0]-points[:,0]<=x_margin)
    near_y=(points[:,1]-lo[1]<=y_margin)|(hi[1]-points[:,1]<=y_margin)
    return float(np.mean(near_x))>=0.55 or float(np.mean(near_y))>=0.55


def _smooth_support_mapping(X,Y,F,W,iterations=4):
    X=np.asarray(X,dtype=np.float64);D=np.asarray(Y,dtype=np.float64)-X;W=np.asarray(W,dtype=np.float64)
    neighbours=[set() for _ in range(len(X))]
    for a,b,c in np.asarray(F,dtype=np.int64):
        a=int(a);b=int(b);c=int(c);neighbours[a].update((b,c));neighbours[b].update((a,c));neighbours[c].update((a,b))
    for _ in range(int(iterations)):
        updated=D.copy()
        for row,items in enumerate(neighbours):
            if len(items)<2:continue
            ids=np.fromiter(items,dtype=np.int64);alignment=W[ids]@W[row]
            floor=max(0.15,float(np.percentile(alignment,25)));similar=ids[alignment>=floor]
            if len(similar)<2:similar=ids
            local=np.median(D[similar],axis=0);residual=float(np.linalg.norm(D[row]-local))
            strength=0.0 if residual<0.012 else (0.12 if residual<0.025 else 0.35)
            updated[row]=(1.0-strength)*D[row]+strength*local
        D=updated
    return X+D


def _filter_support_relief(X,Y,F,W):
    X=np.asarray(X,dtype=np.float64);Y=np.asarray(Y,dtype=np.float64);F=np.asarray(F,dtype=np.int64);W=np.asarray(W,dtype=np.float64)
    if len(X)<4 or len(F)==0:return Y,{"enabled":False,"reason":"insufficient support surface"}
    normals=_vertex_normals(X,F);displacement=Y-X;normal_displacement=np.einsum("ij,ij->i",displacement,normals)
    span=np.ptp(X,axis=0);radius=float(np.clip(np.max(span)*0.035,0.022,0.045));k=min(72,len(X));distance,index=cKDTree(X).query(X,k=k,distance_upper_bound=radius)
    if distance.ndim==1:distance=distance[:,None];index=index[:,None]
    valid=index<len(X);safe=np.where(valid,index,0);bone_alignment=np.einsum("nkj,nj->nk",W[safe],W,optimize=True);normal_alignment=np.einsum("nkj,nj->nk",normals[safe],normals,optimize=True)
    usable=valid&(bone_alignment>=0.18)&(normal_alignment>=0.55);local=np.empty(len(X),dtype=np.float64)
    for row in range(len(X)):
        values=normal_displacement[safe[row,usable[row]]];local[row]=float(np.median(values)) if len(values)>=3 else normal_displacement[row]
    residual=normal_displacement-local;magnitude=np.abs(residual);active=magnitude>.00125;requested=np.zeros(len(X),dtype=np.float64);requested[active]=(local[active]-normal_displacement[active])*1.25
    adjustment_scalar=np.clip(requested,-.010,.010);adjustment=adjustment_scalar[:,None]*normals;output=Y+adjustment;amount=np.linalg.norm(adjustment,axis=1)
    return output,{"enabled":True,"radius_mm":radius*1000.0,"adjusted_vertices":int(np.count_nonzero(amount>1e-6)),"adjustment_p50_mm":float(np.percentile(amount,50)*1000.0),"adjustment_p95_mm":float(np.percentile(amount,95)*1000.0),"adjustment_max_mm":float(np.max(amount)*1000.0),"normal_residual_p95_mm":float(np.percentile(magnitude,95)*1000.0),"clipped_vertices":int(np.count_nonzero(np.abs(requested)>.010))}


def _surface_quality(X,Y,F):
    F=np.asarray(F,dtype=np.int64)
    if len(F)==0:return {'edge_stretch_p95':0.0,'edge_stretch_p99':0.0,'edge_stretch_over_3x_fraction':0.0,'orientation_flip_fraction':0.0,'orientation_p01':1.0}
    edges=np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]));edges=np.unique(np.sort(edges,axis=1),axis=0)
    source=np.linalg.norm(X[edges[:,0]]-X[edges[:,1]],axis=1);target=np.linalg.norm(Y[edges[:,0]]-Y[edges[:,1]],axis=1);ratio=target/np.maximum(source,1e-9)
    source_normals=np.cross(X[F[:,1]]-X[F[:,0]],X[F[:,2]]-X[F[:,0]]);target_normals=np.cross(Y[F[:,1]]-Y[F[:,0]],Y[F[:,2]]-Y[F[:,0]])
    denom=np.maximum(np.linalg.norm(source_normals,axis=1)*np.linalg.norm(target_normals,axis=1),1e-12);alignment=np.einsum('ij,ij->i',source_normals,target_normals)/denom
    return {'edge_stretch_p95':float(np.percentile(ratio,95)),'edge_stretch_p99':float(np.percentile(ratio,99)),'edge_stretch_over_3x_fraction':float(np.mean(ratio>3.0)),'orientation_flip_fraction':float(np.mean(alignment<0.0)),'orientation_p01':float(np.percentile(alignment,1))}


def _support_pair(source_ref,Y):
    X=np.asarray(source_ref['V'],dtype=np.float64);F=np.asarray(source_ref['F'],dtype=np.int64);W=np.asarray(source_ref['W'],dtype=np.float64)
    WX,WF,WY,WW,inverse=_weld_support(X,F,Y,W);raw_quality=_surface_quality(WX,WY,WF);WY=_smooth_support_mapping(WX,WY,WF,WW);smoothed_quality=_surface_quality(WX,WY,WF);relief_Y,relief_stats=_filter_support_relief(WX,WY,WF,WW);relief_C=(relief_Y-WY)[inverse]
    source_normals=_vertex_normals(WX,WF);target_normals=_vertex_normals(WY,WF);repaired_Y=WY[inverse];repaired_NS=source_normals[inverse];repaired_NT=target_normals[inverse]
    groups=_boundary_groups(WX,WF);filled=0;outer=0;faces=[WF];source_vertices=[WX];target_vertices=[WY];base=len(WX)
    cap_source=[];cap_target=[];cap_faces=[]
    for ids,edges in groups:
        if len(ids)<3:continue
        if _outer_boundary(WX,ids):outer+=1;continue
        source_centre=np.mean(WX[ids],axis=0);target_centre=np.mean(WY[ids],axis=0);centre=base+len(cap_source)
        cap_source.append(source_centre);cap_target.append(target_centre)
        for a,b in edges:cap_faces.append((int(b),int(a),centre))
        filled+=1
    if cap_source:
        source_vertices.append(np.asarray(cap_source,dtype=np.float64));target_vertices.append(np.asarray(cap_target,dtype=np.float64));faces.append(np.asarray(cap_faces,dtype=np.int64))
    source_V=np.vstack(source_vertices);target_V=np.vstack(target_vertices);support_F=np.vstack(faces)
    return source_V,support_F,target_V,support_F.copy(),repaired_Y,repaired_NS,repaired_NT,relief_C,{
        'welded_vertices':int(len(WX)),'boundary_groups':int(len(groups)),'outer_boundaries_retained':int(outer),'internal_holes_filled':int(filled),
        'raw_mapping_quality':raw_quality,'support_mapping_quality':smoothed_quality,'relief_filter':relief_stats,
        'mapping_adjustment_p95_mm':float(np.percentile(np.linalg.norm(repaired_Y-np.asarray(Y,dtype=np.float64),axis=1),95)*1000.0),
    }


def build_dense_source_proxy(source_ref,target_ref,k=32):
    """Build a dense source-body support proxy in the target topology.

    Vanilla XIV bodies are intentionally low-poly compared with modern modded bodies.  Feeding
    that sparse topology directly into the body-field builder makes the subsequent garment solve
    depend on a coarse cross-topology correspondence.  Instead, first solve the sparse vanilla ->
    target body motion, interpolate that smooth body displacement onto every target vertex, and
    subtract it from the target.  The resulting proxy represents the vanilla body shape while
    sharing the target's dense topology/UV/rig space exactly.

    The literal source body is retained separately for collision/replacement/removal evidence.
    Modded/RBODY sources never enter this helper.
    """
    if source_ref['joint_names']!=target_ref['joint_names']:
        raise ValueError('Dense source proxy requires source and target in the same rig joint list')
    sparse=collect_slot_pair(source_ref,target_ref)
    X=np.asarray(sparse['X'],dtype=np.float64);Y=np.asarray(sparse['Y'],dtype=np.float64);D=Y-X;BW=np.asarray(sparse['BW'],dtype=np.float64)
    TV=np.asarray(target_ref['V'],dtype=np.float64);TW=np.asarray(target_ref['W'],dtype=np.float64)
    if len(X)<8 or len(TV)<8:raise ValueError('Dense source proxy requires non-empty source and target body surfaces')
    kk=min(max(int(k),8),len(X));tree=cKDTree(Y);distance,index=tree.query(TV,k=kk)
    if distance.ndim==1:distance=distance[:,None];index=index[:,None]
    alignment=np.einsum('nkj,nj->nk',BW[index],TW,optimize=True)
    spatial_scale=max(.008,float(np.percentile(distance[:,0],90))*2.0)
    score=(distance/max(spatial_scale,1e-9))+(2.5*(1.0-alignment))
    source_candidates=X[index]
    crossed=(np.sign(source_candidates[:,:,0])!=np.sign(TV[:,None,0]))&(np.abs(source_candidates[:,:,0])>.010)&(np.abs(TV[:,None,0])>.025)
    score+=crossed.astype(np.float64)*2.0
    base=np.min(score,axis=1,keepdims=True);blend=np.exp(-(score-base)*2.0);blend/=np.maximum(np.sum(blend,axis=1,keepdims=True),1e-12)
    dense_displacement=np.sum(D[index]*blend[:,:,None],axis=1);PV=TV-dense_displacement;PF=np.asarray(target_ref['F'],dtype=np.int64).copy();PN=_vertex_normals(PV,PF)
    proxy={key:(value.copy() if hasattr(value,'copy') else value) for key,value in target_ref.items()}
    proxy.update({
        'V':PV,'N':PN,'F':PF,'W':TW.copy(),'UV':np.asarray(target_ref['UV'],dtype=np.float64).copy(),'joint_names':list(target_ref['joint_names']),
        'payload_id':f"dense-vanilla:{source_ref.get('payload_id','source')}->{target_ref.get('payload_id','target')}",
        'slot':source_ref.get('slot',target_ref.get('slot','Body')),'race_code':source_ref.get('race_code'),'surface_mode':'body',
        '_dense_exact_target_topology':True,
        '_literal_source_V':np.asarray(source_ref['V'],dtype=np.float64).copy(),
        '_literal_source_F':np.asarray(source_ref['F'],dtype=np.int64).copy(),
        '_literal_source_W':np.asarray(source_ref['W'],dtype=np.float64).copy(),
        '_literal_source_mesh_records':[dict(row) for row in source_ref.get('mesh_records',[])],
        '_dense_proxy_stats':{
            'mode':'dense_vanilla_target_topology','sparse_source_vertices':int(len(X)),'dense_proxy_vertices':int(len(PV)),
            'seed_correspondence_mode':str(sparse['stats'].get('correspondence_mode','unknown')),
            'seed_uv_fallback_fraction':float(sparse['stats'].get('uv_fallback_fraction',0.0)),
            'seed_spatial_p95_mm':float(sparse['stats'].get('chosen_spatial_p95_mm',0.0)),
            'dense_field_rms_mm':float(np.sqrt(np.mean(np.sum(dense_displacement*dense_displacement,axis=1)))*1000.0),
            'interpolation_k':int(kk),'interpolation_spatial_scale_mm':float(spatial_scale*1000.0),
        },
    })
    return proxy



def apply_embedded_source_authority(canonical_ref,embedded_ref):
    """Use the body surface exported with a mod as local source-fit geometry.

    The RBODY selection remains semantic/completion authority: topology, UVs, rig space and any
    source regions deleted by the outfit still come from the catalogue.  Where the outfit really
    contains body geometry, that literal surface wins geometrically so authored garment clearance
    is measured against the body the modder actually fitted to, not a nearby catalogue revision.
    """
    if list(canonical_ref.get('joint_names',[]))!=list(embedded_ref.get('joint_names',[])):
        raise ValueError('Embedded and catalogue source bodies are not in the same rig joint space')
    X=np.asarray(canonical_ref['V'],dtype=np.float64);F=np.asarray(canonical_ref['F'],dtype=np.int64)
    E=np.asarray(embedded_ref['V'],dtype=np.float64);EF=np.asarray(embedded_ref['F'],dtype=np.int64)
    if len(X)<8 or len(F)==0 or len(E)<8 or len(EF)==0:
        return canonical_ref,{'enabled':False,'reason':'insufficient embedded/catalogue body geometry'}

    out=dict(canonical_ref)
    out['V']=X.copy();out['F']=F.copy();out['UV']=np.asarray(canonical_ref['UV'],dtype=np.float64).copy();out['W']=np.asarray(canonical_ref['W'],dtype=np.float64).copy()
    canonical_N=_unit(np.asarray(canonical_ref['N'],dtype=np.float64));embedded_N=_unit(np.asarray(embedded_ref['N'],dtype=np.float64))
    out['N']=canonical_N.copy()
    original_payload=str(canonical_ref.get('payload_id') or 'catalogue-source')

    exact_topology=len(X)==len(E) and F.shape==EF.shape and np.array_equal(F,EF)
    if exact_topology:
        delta=np.linalg.norm(E-X,axis=1)
        out['V']=E.copy();out['N']=embedded_N.copy()
        alpha=np.ones(len(X),dtype=np.float64)
        mode='exact-embedded-topology'
        diag={'chosen_spatial':delta,'chosen_alignment':np.ones(len(X),dtype=np.float64),'uv_nearest':np.zeros(len(X),dtype=np.float64)}
    else:
        mapped,mapped_normals,diag=uv_correspondence(canonical_ref,embedded_ref,k=48)
        mapped=np.asarray(mapped,dtype=np.float64);mapped_normals=_unit(np.asarray(mapped_normals,dtype=np.float64))
        spatial=np.asarray(diag.get('chosen_spatial',np.full(len(X),np.inf)),dtype=np.float64)
        alignment=np.asarray(diag.get('chosen_alignment',np.zeros(len(X))),dtype=np.float64)
        uv=np.asarray(diag.get('uv_nearest',np.full(len(X),np.nan)),dtype=np.float64)
        body_diagonal=max(float(np.linalg.norm(np.ptp(X,axis=0))),1e-6)
        full_distance=float(np.clip(body_diagonal*0.0035,0.0015,0.0030))
        reject_distance=float(np.clip(body_diagonal*0.014,0.0080,0.0140))
        max_displacement=float(np.clip(body_diagonal*0.035,0.018,0.032))

        # RBODY variants commonly share most source vertices exactly even when the outfit deleted
        # a few hidden regions.  Preserve those literal vertices exactly rather than averaging a
        # correspondence neighbourhood (which can falsely soften anatomy by 1-2 mm).
        embedded_W=np.asarray(embedded_ref['W'],dtype=np.float64);tree=cKDTree(E);nearest_distance,nearest_index=tree.query(X,k=1)
        nearest_index=np.asarray(nearest_index,dtype=np.int64);nearest_distance=np.asarray(nearest_distance,dtype=np.float64)
        nearest_alignment=np.einsum('ij,ij->i',embedded_W[nearest_index],np.asarray(canonical_ref['W'],dtype=np.float64),optimize=True)
        literal_distance=float(np.clip(body_diagonal*0.0010,0.00045,0.00080))
        literal=(nearest_distance<=literal_distance)&(nearest_alignment>=0.42)
        mapped[literal]=E[nearest_index[literal]];mapped_normals[literal]=embedded_N[nearest_index[literal]]

        displacement=np.linalg.norm(mapped-X,axis=1)
        alpha=np.clip((reject_distance-spatial)/max(reject_distance-full_distance,1e-9),0.0,1.0)
        rig_alpha=np.clip((alignment-0.28)/0.52,0.0,1.0);alpha*=rig_alpha
        # Non-literal correspondence is deliberately only a bridge into the missing-region
        # completion.  Literal source samples are the authority; uncertain holes remain RBODY.
        alpha[~literal]*=0.35;alpha[literal]=1.0
        bad_uv=np.isfinite(uv)&(uv>0.075)&(spatial>full_distance);alpha[bad_uv&~literal]*=0.20
        alpha[(spatial>reject_distance)|(alignment<0.28)|(displacement>max_displacement)]=0.0
        alpha[literal]=1.0
        coverage=float(np.mean(alpha>=0.20))
        if coverage<0.12:
            return canonical_ref,{
                'enabled':False,'reason':'embedded body does not cover enough of this selected slot','mode':'catalogue-fallback',
                'coverage_fraction':coverage,'embedded_vertices':int(len(E)),'catalogue_vertices':int(len(X)),
            }
        out['V']=X+(mapped-X)*alpha[:,None]
        blended=(canonical_N*(1.0-alpha[:,None]))+(mapped_normals*alpha[:,None]);out['N']=_unit(blended)
        delta=np.linalg.norm(out['V']-X,axis=1)
        mode='hybrid-embedded-with-rbody-completion'

    # Once literal embedded geometry changes X this must not be mistaken for an identity RBODY map.
    out['_canonical_payload_id']=original_payload
    out['payload_id']=f'embedded-local:{original_payload}'
    out['_literal_source_V']=np.asarray(out['V'],dtype=np.float64).copy()
    out['_literal_source_F']=F.copy()
    out['_literal_source_W']=np.asarray(out['W'],dtype=np.float64).copy()
    out['_literal_source_mesh_records']=[dict(row) for row in canonical_ref.get('mesh_records',[])]
    report={
        'enabled':True,'mode':mode,'catalogue_payload':original_payload,'catalogue_vertices':int(len(X)),'embedded_vertices':int(len(E)),
        'authority_fraction':float(np.mean(alpha>=0.20)),'full_authority_fraction':float(np.mean(alpha>=0.95)),
        'moved_vertices':int(np.count_nonzero(delta>1e-6)),'move_p50_mm':float(np.percentile(delta,50)*1000.0),
        'move_p95_mm':float(np.percentile(delta,95)*1000.0),'move_max_mm':float(np.max(delta)*1000.0),
        'policy':'embedded source body is local geometric authority; RBODY supplies semantics/topology and missing-region completion',
    }
    if not exact_topology:
        spatial=np.asarray(diag.get('chosen_spatial',np.zeros(len(X))),dtype=np.float64);alignment=np.asarray(diag.get('chosen_alignment',np.ones(len(X))),dtype=np.float64)
        report.update({
            'correspondence_mode':str(diag.get('mode','unknown')),'chosen_spatial_p95_mm':float(np.percentile(spatial,95)*1000.0),
            'chosen_alignment_p50':float(np.median(alignment)),'literal_vertex_authority_fraction':float(np.mean(literal)),'completion_fraction':float(np.mean(alpha<0.20)),
        })
    out['_embedded_source_authority']=dict(report)
    return out,report

def collect_slot_pair(source_ref,target_ref):
    if source_ref['joint_names']!=target_ref['joint_names']:raise ValueError('Source and target references are not remapped to the same rig joint list')
    dense_exact=bool(source_ref.get('_dense_exact_target_topology',False))
    if dense_exact:
        if len(source_ref['V'])!=len(target_ref['V']) or not np.array_equal(np.asarray(source_ref['F'],dtype=np.int64),np.asarray(target_ref['F'],dtype=np.int64)):
            raise ValueError('Dense exact-topology source proxy no longer matches the selected target topology')
        raw_Y=np.asarray(target_ref['V'],dtype=np.float64).copy();diag={'uv_nearest':np.zeros(len(raw_Y),dtype=np.float64),'chosen_spatial':np.zeros(len(raw_Y),dtype=np.float64),'chosen_alignment':np.ones(len(raw_Y),dtype=np.float64),'spatial_scale':0.0,'mode':'dense_exact_topology','hybrid_quality':{},'spatial_quality':{}}
    else:
        if bool(source_ref.get('_cross_sex_smallclothes_bridge',False)):
            raw_Y,_,diag=_cross_sex_rig_correspondence(source_ref,target_ref)
        else:
            raw_Y,_,diag=uv_correspondence(source_ref,target_ref)
    X=np.asarray(source_ref['V'],dtype=np.float64);BW=np.asarray(source_ref['W'],dtype=np.float64);good=BW.sum(1)>1e-8
    if not np.all(good):raise ValueError(f'{np.count_nonzero(~good)} source body vertices have no weights after rig remap')
    source_support_V,source_support_F,target_support_V,target_support_F,Y,NS,NT,relief_C,support_stats=_support_pair(source_ref,raw_Y)
    uvd=diag['uv_nearest'];chosen_spatial=diag['chosen_spatial'];chosen_alignment=diag['chosen_alignment']
    identity_payload=bool(source_ref.get('payload_id')) and source_ref.get('payload_id')==target_ref.get('payload_id') and str(source_ref.get('race_code') or '')==str(target_ref.get('race_code') or '') and str(source_ref.get('surface_mode') or 'body').casefold()=='body' and str(target_ref.get('surface_mode') or 'body').casefold()=='body'
    return {
        'X':X,'Y':Y,'BW':BW,'NS':NS,'NT':NT,'names':list(source_ref['joint_names']),'parts':np.asarray([source_ref.get('slot','Body')]*len(X),dtype=object),'identity_payload':identity_payload,
        'source_support_V':source_support_V,'source_support_F':source_support_F,'target_support_V':target_support_V,'target_support_F':target_support_F,'target_relief_C':relief_C,
        'source_literal_V':np.asarray(source_ref.get('_literal_source_V',source_ref['V']),dtype=np.float64),'source_literal_F':np.asarray(source_ref.get('_literal_source_F',source_ref['F']),dtype=np.int64),'source_literal_W':np.asarray(source_ref.get('_literal_source_W',source_ref['W']),dtype=np.float64),
        'target_literal_V':np.asarray(target_ref.get('_cross_sex_collision_V',target_ref['V']),dtype=np.float64),'target_literal_F':np.asarray(target_ref.get('_cross_sex_collision_F',target_ref['F']),dtype=np.int64),'target_literal_W':np.asarray(target_ref.get('_cross_sex_collision_W',target_ref['W']),dtype=np.float64),
        'source_mesh_records':[dict(row) for row in source_ref.get('_literal_source_mesh_records',source_ref.get('mesh_records',[]))],'target_mesh_records':[dict(row) for row in target_ref.get('mesh_records',[])],
        'slot':source_ref.get('slot','Body'),
        'stats':{
            'slot':source_ref.get('slot','Body'),'source_vertices':int(len(X)),'target_vertices':int(len(target_ref['V'])),
            'uv_p50':float(np.median(uvd)),'uv_p95':float(np.percentile(uvd,95)),'uv_fallback_fraction':float(np.mean(uvd>=0.025)),
            'chosen_spatial_p95_mm':float(np.percentile(chosen_spatial,95)*1000.0),'chosen_alignment_p50':float(np.median(chosen_alignment)),
            'spatial_scale_mm':float(diag['spatial_scale']*1000.0),'correspondence_mode':str(diag.get('mode','uv_rig_spatial')),'correspondence_hybrid_quality':dict(diag.get('hybrid_quality',{})),'correspondence_spatial_quality':dict(diag.get('spatial_quality',{})),'displacement_rms_mm':float(np.sqrt(np.mean(np.sum((Y-X)**2,axis=1)))*1000),
            'raw_displacement_rms_mm':float(np.sqrt(np.mean(np.sum((raw_Y-X)**2,axis=1)))*1000),
            'support_proxy':support_stats,
            'dense_source_proxy':dict(source_ref.get('_dense_proxy_stats',{})) if dense_exact else None,
            'cross_sex_collision_mode':str(target_ref.get('_cross_sex_collision_mode') or 'literal_target_surface'),
        },
    }


def _combine(rows,v_key,f_key,w_key=None):
    vertices=[];faces=[];weights=[];base=0;weight_width=None
    for row in rows:
        V=np.asarray(row[v_key],dtype=np.float64);F=np.asarray(row[f_key],dtype=np.int64);vertices.append(V);faces.append(F+base);base+=len(V)
        if w_key is not None:
            W=np.asarray(row[w_key],dtype=np.float64)
            if weight_width is None:weight_width=W.shape[1]
            elif W.shape[1]!=weight_width:raise ValueError('Body surface references do not share one rig-weight space')
            weights.append(W)
    V=np.vstack(vertices) if vertices else np.zeros((0,3),np.float64);F=np.vstack(faces) if faces else np.zeros((0,3),np.int64)
    if w_key is None:return V,F
    W=np.vstack(weights) if weights else np.zeros((0,0 if weight_width is None else weight_width),np.float64);return V,F,W


def collect_body_pairs(slot_pairs):
    """Combine selected body regions into B14 body-cache arrays."""
    pairs=[collect_slot_pair(s,t) for s,t in slot_pairs]
    if not pairs:raise ValueError('No body slot pairs supplied')
    names=pairs[0]['names']
    if any(p['names']!=names for p in pairs[1:]):raise ValueError('All slot pairs must use the same rig joint order')
    source_literal_V,source_literal_F,source_literal_W=_combine(pairs,'source_literal_V','source_literal_F','source_literal_W')
    target_literal_V,target_literal_F,target_literal_W=_combine(pairs,'target_literal_V','target_literal_F','target_literal_W')
    source_support_V,source_support_F=_combine(pairs,'source_support_V','source_support_F')
    target_support_V,target_support_F=_combine(pairs,'target_support_V','target_support_F')
    return {
        'X':np.vstack([p['X'] for p in pairs]),'Y':np.vstack([p['Y'] for p in pairs]),'BW':np.vstack([p['BW'] for p in pairs]),
        'NS':np.vstack([p['NS'] for p in pairs]),'NT':np.vstack([p['NT'] for p in pairs]),'names':names,'parts':np.concatenate([p['parts'] for p in pairs]),
        'slot_stats':[p['stats'] for p in pairs],'slot_pairs':pairs,
        'source_surface_V':source_literal_V,'source_surface_F':source_literal_F,'source_surface_W':source_literal_W,
        'target_surface_V':target_literal_V,'target_surface_F':target_literal_F,'target_surface_W':target_literal_W,
        'source_support_V':source_support_V,'source_support_F':source_support_F,'target_support_V':target_support_V,'target_support_F':target_support_F,
        'target_relief_C':np.vstack([p['target_relief_C'] for p in pairs]),
        'identity_body_mapping':bool(all(bool(p.get('identity_payload')) for p in pairs)),
        'dense_vanilla_source_proxy':bool(all(str((p.get('stats',{}).get('dense_source_proxy') or {}).get('mode') or '')=='dense_vanilla_target_topology' for p in pairs)),
        'dense_cross_sex_source_proxy':bool(any(str((p.get('stats',{}).get('dense_source_proxy') or {}).get('mode') or '')=='dense_cross_sex_target_topology' for p in pairs)),
    }
