from __future__ import annotations
import numpy as np
import final_occupancy_guard as guard

class _Source:
    def __init__(self):self._data={"mesh":{"V":np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),"F":np.asarray([[0,1,2]],dtype=np.int64)}}
    def data(self,name):return self._data[name]
class _Prod:pass

def test_dense_face_sampling_catches_interior_witness_vertices_and_centroid_miss():
    prod=_Prod()
    def occupancy(points,triangles,k=48,exact_band=.002):
        points=np.asarray(points,float);hit=(np.abs(points[:,0]-.5)<1e-10)&(np.abs(points[:,1]-.25)<1e-10);signed=np.where(hit,-.001,.002);return points,np.tile([[0.,0.,1.]],(len(points),1)),signed,np.abs(signed),np.zeros(len(points),dtype=np.int64)
    prod._nearest_literal_occupancy=occupancy;source=_Source();report=guard._dense_penetration_report(prod,source,{"mesh":source.data("mesh")["V"].copy()},np.zeros((1,3,3)),.00012);assert report["penetrating_samples"]==1;assert report["meshes"][0]["samples"]==16

def test_dense_finalizer_refuses_residual_body_poke():
    prod=_Prod();source=_Source();base={"mesh":source.data("mesh")["V"].copy()};prod._finalize_modded_coupled_solution=lambda source,cache,positions,skinning,records,contexts,rms:(positions,skinning,records,{});prod._target_fit_collision_triangles=lambda cache:np.zeros((1,3,3));prod._triangles_from_surface=lambda v,f:np.asarray(v)[np.asarray(f)];prod._final_target_body_clearance=lambda source,positions,target,support,**kwargs:(positions,{})
    def occupancy(points,triangles,k=48,exact_band=.002):
        signed=np.full(len(points),-.001);return points,np.tile([[0.,0.,1.]],(len(points),1)),signed,np.abs(signed),np.zeros(len(points),dtype=np.int64)
    prod._nearest_literal_occupancy=occupancy;cache={"target_support_V":np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),"target_support_F":np.asarray([[0,1,2]],dtype=np.int64)};guard.install_dense_final_target_occupancy(prod)
    try:prod._finalize_modded_coupled_solution(source,cache,base,{}, {}, {},0.)
    except ValueError as ex:assert "refused to emit" in str(ex)
    else:raise AssertionError("Residual target-body penetration must fail conversion")
