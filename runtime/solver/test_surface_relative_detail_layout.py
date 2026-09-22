from __future__ import annotations
import numpy as np
from surface_relative_detail_layout import SurfaceRelativeDetailConfig,preserve_surface_relative_detail_layout


def _grid(nx=12,ny=8,dx=.01,dy=.01):
    V=np.asarray([[i*dx,j*dy,0.0] for j in range(ny) for i in range(nx)],dtype=np.float64);F=[]
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;b=a+1;c=a+nx;d=c+1;F.extend(((a,b,c),(b,d,c)))
    return V,np.asarray(F,dtype=np.int64)


def _cube(cx,cy,cz=.004,s=.006):
    a=s/2;V=np.asarray([[cx+x,cy+y,cz+z] for x,y,z in [(-a,-a,-a),(a,-a,-a),(a,a,-a),(-a,a,-a),(-a,-a,a),(a,-a,a),(a,a,a),(-a,a,a)]],dtype=np.float64)
    F=np.asarray([(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),(1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0)],dtype=np.int64);return V,F


def test_disconnected_detail_inherits_solved_carrier_frame_instead_of_independent_drift():
    host,hf=_grid();detail,df=_cube(.055,.035);source=np.vstack((host,detail));faces=np.vstack((hf,df+len(host)))
    angle=.12;R=np.asarray([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]],dtype=np.float64);scale=.9;t=np.asarray([.02,-.01,.003])
    solved=source.copy();solved[:len(host)]=scale*(host@R)+t;solved[len(host):]=scale*(detail@R)+t+np.asarray([.006,-.004,.002])
    cfg=SurfaceRelativeDetailConfig(min_host_vertices=50,min_host_extent_m=.04,min_detail_vertices=8,max_detail_vertices=32,max_source_median_clearance_m=.02,max_source_p90_clearance_m=.03)
    out,report=preserve_surface_relative_detail_layout(source,solved,faces,config=cfg)
    expected=scale*(detail@R)+t
    assert report['corrected_detail_count']==1 and report['moved_vertices']==len(detail)
    assert np.max(np.linalg.norm(out[len(host):]-expected,axis=1))<1e-8
    assert np.array_equal(out[:len(host)],solved[:len(host)])


def test_far_compact_component_is_not_assumed_to_be_carried_detail():
    host,hf=_grid();detail,df=_cube(.30,.30);source=np.vstack((host,detail));faces=np.vstack((hf,df+len(host)));solved=source.copy();solved[:len(host),0]*=.9;solved[len(host):]+=np.asarray([.01,0,0])
    cfg=SurfaceRelativeDetailConfig(min_host_vertices=50,min_host_extent_m=.04,min_detail_vertices=8,max_detail_vertices=32,max_source_median_clearance_m=.01,max_source_p90_clearance_m=.015)
    out,report=preserve_surface_relative_detail_layout(source,solved,faces,config=cfg)
    assert report['corrected_detail_count']==0 and np.array_equal(out,solved)
