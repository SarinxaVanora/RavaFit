from __future__ import annotations

import numpy as np

from source_relative_structural_carriers import StructuralCarrierConfig, preserve_source_relative_structural_carriers


def _grid(nx=18, ny=12, dx=.008, dy=.008):
    V=np.asarray([[i*dx,j*dy,0.0] for j in range(ny) for i in range(nx)],dtype=np.float64);F=[]
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;b=a+1;c=a+nx;d=c+1;F.extend(((a,b,c),(b,d,c)))
    return V,np.asarray(F,dtype=np.int64)


def _ring(cx,cy,r=.018,n=24,z=.006):
    # thin triangulated annulus, disconnected from its nested ornament
    pts=[]
    for rr in (r*.72,r):
        for a in np.linspace(0,2*np.pi,n,endpoint=False):pts.append((cx+rr*np.cos(a),cy+rr*np.sin(a),z))
    V=np.asarray(pts,float);F=[]
    for i in range(n):
        j=(i+1)%n;F.extend(((i,j,n+i),(j,n+j,n+i)))
    return V,np.asarray(F,np.int64)


def _star(cx,cy,r=.010,z=.0065):
    pts=[(cx,cy,z)]
    for i,a in enumerate(np.linspace(0,2*np.pi,10,endpoint=False)):
        rr=r if i%2==0 else r*.42;pts.append((cx+rr*np.cos(a),cy+rr*np.sin(a),z))
    V=np.asarray(pts,float);F=[]
    for i in range(10):F.append((0,1+i,1+((i+1)%10)))
    return V,np.asarray(F,np.int64)


def _cuff(cx=.06,cy=.11,rx=.055,rz=.050,height=.018,n=36):
    V=[]
    for yy in (cy-height/2,cy+height/2):
        for a in np.linspace(0,2*np.pi,n,endpoint=False):V.append((cx+rx*np.cos(a),yy,rz*np.sin(a)))
    V=np.asarray(V,float);F=[]
    for i in range(n):
        j=(i+1)%n;F.extend(((i,j,n+i),(j,n+j,n+i)))
    return V,np.asarray(F,np.int64)


def _tube(cx=.06,y0=-.10,y1=.10,rx=.052,rz=.047,rings=12,n=36):
    V=[]
    for y in np.linspace(y0,y1,rings):
        for a in np.linspace(0,2*np.pi,n,endpoint=False):V.append((cx+rx*np.cos(a),y,rz*np.sin(a)))
    V=np.asarray(V,float);F=[]
    for j in range(rings-1):
        for i in range(n):
            ni=(i+1)%n;a=j*n+i;b=j*n+ni;c=(j+1)*n+i;d=(j+1)*n+ni;F.extend(((a,b,c),(b,d,c)))
    return V,np.asarray(F,np.int64)


def test_nested_ornaments_share_one_transform_so_inner_shape_cannot_escape_outer_ring():
    host,hf=_grid();ring,rf=_ring(.075,.045);star,sf=_star(.075,.045)
    source=np.vstack((host,ring,star));faces=np.vstack((hf,rf+len(host),sf+len(host)+len(ring)))
    angle=.10;R=np.asarray([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]],float);scale=.88;t=np.asarray([.015,-.006,.002])
    solved=scale*(source@R)+t
    # Break only the authored nested relationship: move the star outside the ring.
    star_ids=np.arange(len(host)+len(ring),len(source));solved[star_ids]+=np.asarray([.020,-.004,0])
    cfg=StructuralCarrierConfig(assembly_source_gap_m=.008,assembly_max_component_extent_m=.06,assembly_min_component_vertices=8,assembly_max_component_vertices=100)
    out,report=preserve_source_relative_structural_carriers(source,solved,faces,config=cfg)
    ring_ids=np.arange(len(host),len(host)+len(ring));
    assert report['corrected_assembly_count']>=1
    # Source coincidence of ring/star centres remains coincidence after the shared target transform.
    assert np.linalg.norm(out[star_ids].mean(0)-out[ring_ids].mean(0))<.002
    # Host is not owned by ornament authority.
    assert np.array_equal(out[:len(host)],solved[:len(host)])


def test_broad_thin_cuff_uses_one_similarity_from_fitted_host_instead_of_bending_independently():
    host,hf=_tube();cuff,cf=_cuff(cy=.105)
    source=np.vstack((host,cuff));faces=np.vstack((hf,cf+len(host)))
    angle=.06;R=np.asarray([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]],float);scale=.86;t=np.asarray([.008,.002,-.004])
    solved=source.copy();solved[:len(host)]=scale*(host@R)+t
    solved[len(host):]=scale*(cuff@R)+t
    # Artificially dome/bend the cuff while leaving the fitted host valid.
    ids=np.arange(len(host),len(source));phase=(source[ids,2]-source[ids,2].min())/max(np.ptp(source[ids,2]),1e-9)
    solved[ids,1]+=.010*np.sin(np.pi*phase)
    cfg=StructuralCarrierConfig(band_host_min_vertices=250,band_anchor_vertices=220,band_source_median_clearance_m=.05,band_source_p90_clearance_m=.07)
    out,report=preserve_source_relative_structural_carriers(source,solved,faces,config=cfg)
    corrected=[r for r in report['bands'] if r.get('status')=='carrier_relative_band_corrected']
    assert corrected
    expected=scale*(cuff@R)+t
    assert np.percentile(np.linalg.norm(out[ids]-expected,axis=1),95)<.003
    assert np.array_equal(out[:len(host)],solved[:len(host)])


def test_band_prefers_valid_carrier_from_its_own_authored_mesh_over_closer_foreign_layer():
    own_host,hf=_tube(rx=.045,rz=.042);cuff,cf=_cuff(cy=.105,rx=.055,rz=.050)
    foreign,ff=_tube(rx=.052,rz=.048)
    source=np.vstack((own_host,cuff,foreign))
    faces=np.vstack((hf,cf+len(own_host),ff+len(own_host)+len(cuff)))
    # The source mesh boundary is explicit evidence: own host + cuff were authored together, while
    # the geometrically closer foreign shell is a neighbouring layer from another source mesh.
    groups=np.concatenate((np.zeros(len(own_host)+len(cuff),dtype=np.int64),np.ones(len(foreign),dtype=np.int64)))
    a=.05;R=np.asarray([[np.cos(a),0,-np.sin(a)],[0,1,0],[np.sin(a),0,np.cos(a)]],float);scale=.86;t=np.asarray([.008,.002,-.004])
    solved=source.copy();solved[:len(own_host)]=scale*(own_host@R)+t
    off=len(own_host)+len(cuff);solved[off:]=.96*foreign+np.asarray([.020,-.004,.010])
    ids=np.arange(len(own_host),len(own_host)+len(cuff));solved[ids]=scale*(cuff@R)+t
    solved[ids,1]+=.009*np.sin(np.linspace(0,np.pi,len(ids)))
    cfg=StructuralCarrierConfig(band_host_min_vertices=250,band_anchor_vertices=220,band_source_median_clearance_m=.05,band_source_p90_clearance_m=.07)
    out,report=preserve_source_relative_structural_carriers(source,solved,faces,config=cfg,vertex_group_ids=groups)
    expected=scale*(cuff@R)+t
    assert np.percentile(np.linalg.norm(out[ids]-expected,axis=1),95)<.003
    corrected=[r for r in report['bands'] if r.get('status')=='carrier_relative_band_corrected']
    assert corrected
    # The chosen carrier must be the own-mesh host (component zero in this synthetic source).
    assert any(int(r.get('host_component',-1))==0 for r in corrected)
