from __future__ import annotations

import numpy as np

import local_support_preserve as preserve


class _Source:
    def __init__(self, meshes): self._meshes=meshes
    def data(self,name): return self._meshes[name]


class _Prod:
    pass


def _mesh(z):
    return {
        "V":np.asarray([[0.,0.,z],[.01,0.,z],[0.,.01,z]],dtype=np.float64),
        "F":np.asarray([[0,1,2]],dtype=np.int64),
        "W":np.asarray([[.7,.3],[.6,.4],[.5,.5]],dtype=np.float64),
        "joint_names":["j_a","j_b"],
    }


def _fake_prod(meshes):
    prod=_Prod()
    def original(source,cache,positions,skinning,records,contexts,rms):
        out={name:np.asarray(data["V"])+np.asarray([.003,0.,.002]) for name,data in meshes.items()}
        skins={name:{"weights":np.asarray(data["W"])+np.asarray([[.05,-.05]]*3),"joint_names":data["joint_names"],"stage":{}} for name,data in meshes.items()}
        return out,skins,{name:{} for name in meshes},{"original":True}
    def support_frame(points,cache):
        points=np.asarray(points,dtype=np.float64);changed=points[:,2]>5.0;contact=points.copy();contact[changed,0]+=.020
        return contact,np.tile([[0.,0.,1.]],(len(points),1)),np.full(len(points),.001),np.zeros(len(points),dtype=np.int64)
    def nearest(points,triangles,k=48):
        points=np.asarray(points,dtype=np.float64)
        return points.copy(),np.tile([[0.,0.,1.]],(len(points),1)),np.zeros(len(points)),np.zeros(len(points)),np.zeros(len(points),dtype=np.int64)
    def delta(points,weights,names,cache):
        points=np.asarray(points,dtype=np.float64);local=np.where(points[:,2]>5.0,.25,0.0)
        return np.zeros((len(points),2)),{"local_delta_l1":local,"quantisation_floor":.0035}
    prod._finalize_modded_coupled_solution=original
    prod._coupled_support_frame=support_frame
    prod._b14_nearest_surface=nearest
    prod._triangles_from_surface=lambda v,f: np.asarray(v)[np.asarray(f,dtype=np.int64)]
    prod._verified_body_skin_delta_at_points=delta
    prod.weld_mesh=lambda data:{"raw_to_weld":np.arange(len(data["V"]),dtype=np.int64)}
    return prod


def test_unchanged_leg_is_exactly_preserved_when_chest_changes():
    meshes={"leg":_mesh(0.0),"chest":_mesh(10.0)};source=_Source(meshes);prod=_fake_prod(meshes)
    preserve.install_local_support_preservation(prod)
    positions,skinning,records,stats=prod._finalize_modded_coupled_solution(source,{"source_support_V":meshes["leg"]["V"],"source_support_F":meshes["leg"]["F"]},{},{},{},{},0.0)
    np.testing.assert_array_equal(positions["leg"],meshes["leg"]["V"])
    np.testing.assert_array_equal(skinning["leg"]["weights"],meshes["leg"]["W"])
    assert not np.array_equal(positions["chest"],meshes["chest"]["V"])
    assert stats["local_support_source_authority"]["changed_mesh_count"]==1
    assert records["leg"]["local_support_source_authority"]["exact_source_vertices"]==3


def test_transition_keeps_changed_region_free_and_stable_core_exact():
    faces=np.asarray([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7]],dtype=np.int64)
    stable=np.asarray([False,True,True,True,True,True,True,True])
    alpha=preserve._transition_alpha(faces,stable,np.arange(8),rings=4)
    assert alpha[0]==0.0
    assert alpha[-1]==1.0
    assert np.all(np.diff(alpha)>=-1e-12)


def test_welded_render_duplicates_share_preservation_authority():
    faces=np.asarray([[0,2,3],[1,2,3]],dtype=np.int64);stable=np.asarray([True,False,True,True]);weld=np.asarray([0,0,1,2])
    alpha=preserve._transition_alpha(faces,stable,weld,rings=4)
    assert alpha[0]==alpha[1]==0.0
