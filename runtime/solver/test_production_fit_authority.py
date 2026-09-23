import numpy as np
import pytest
from collections import OrderedDict

import production_b14 as p


def test_selected_rbody_is_not_reshaped_by_embedded_outfit_body(monkeypatch):
    canonical=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    class Library:
        def reference(self,*args,**kwargs):
            return {"V":canonical.copy(),"payload_id":"selected-body"}
        def payload_meta(self,*args):return {"body_surface_materials":["/body.mtrl"]}
        def mesh_material_assignments(self,*args):return {0:{"material":"/body.mtrl"}}
    monkeypatch.setattr(p,"_PAIR_CACHE",OrderedDict())
    monkeypatch.setattr(p,"_pair_cache_key",lambda *args:("rbody-source-test",))
    monkeypatch.setattr(p,"_race_retarget_context",lambda *args:None)
    monkeypatch.setattr(p,"get_cached_rbody",lambda *args:Library())
    monkeypatch.setattr(p,"_embedded_body_materials",lambda *args:{"/outfit-body.mtrl"})
    def no_embedded_reference(*args):
        raise AssertionError("An explicit RBODY source must not be replaced by the outfit body")
    monkeypatch.setattr(p,"_embedded_body_reference",no_embedded_reference)
    monkeypatch.setattr(p,"collect_body_pairs",lambda pairs:{"pairs":pairs})
    monkeypatch.setattr(p,"_apply_strict_historical_uv_contract",lambda pairs,cache:cache)
    selection={"rbody":"fixture.rbody","body":"fixture","variant":"medium"}
    spec={"source_body_mode":"rbody","slots":[{"slot":"Chest","source":selection,"target":selection}]}

    _,cache,materials,_,_,_,reports=p._load_pairs(spec,["root"],object())

    np.testing.assert_array_equal(cache["pairs"][0][0]["V"],canonical)
    assert materials=={"/outfit-body.mtrl"}
    assert reports[0]["embedded_source_authority"]["mode"]=="rbody_source_authority"


def test_body_visibility_cutout_does_not_become_fit_collision_surface():
    vertices=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    faces=np.asarray([[0,1,2],[1,3,2]])
    visible=vertices[faces[:1]].copy()
    cache={"target_surface_V":vertices,"target_surface_F":faces,
           "_ravafit_source_body_suppression":{"_collision_triangles":visible,
                                               "native_suppression":[{"triangles":[1]}]}}

    collision=p._target_fit_collision_triangles(cache)

    np.testing.assert_array_equal(collision,vertices[faces])
    # Fitting does not change the separately authored output visibility plan.
    np.testing.assert_array_equal(cache["_ravafit_source_body_suppression"]["_collision_triangles"],visible)
    assert cache["_ravafit_source_body_suppression"]["native_suppression"]==[{"triangles":[1]}]


def test_optional_body_package_surface_does_not_mould_clothing():
    body=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    accessory=body+[0.,0.,.005]
    cache={"target_surface_V":np.vstack([body,accessory]),
           "target_surface_F":np.asarray([[0,1,2],[3,4,5]]),
           "_ravafit_strict_target_surface_V":body,
           "_ravafit_strict_target_surface_F":np.asarray([[0,1,2]])}

    collision=p._target_fit_collision_triangles(cache)

    np.testing.assert_array_equal(collision,body[None])
    np.testing.assert_array_equal(cache["target_surface_V"][3:],accessory)


def test_unchanged_skinning_preserves_authored_accessor_bytes_and_order():
    joints=np.asarray([[3,1,0,2],[2,3,1,0]],dtype=np.uint16)
    values=np.asarray([[.1,.2,.6,.1],[.3,.4,.2,.1]],dtype=np.float32)
    class Editor:
        def accessor(self,index):return [joints,values][index].copy()
        def write_accessor(self,*args):raise AssertionError("Unchanged skinning must not be rewritten")
    dense=np.zeros((2,4))
    for i in range(4):dense[np.arange(2),joints[:,i]]+=values[:,i]
    dense/=dense.sum(axis=1,keepdims=True)

    report=p._write_retargeted_skinning(Editor(),{"attributes":{"JOINTS_0":0,"WEIGHTS_0":1}},dense,"cloth")

    assert report["attributes_preserved"]


def test_seam_cleanup_does_not_recalculate_unchanged_source_weights(monkeypatch):
    weights=np.asarray([[.1,.2,.6,.1],[.3,.4,.2,.1]],dtype=np.float32).astype(np.float64)
    weights/=weights.sum(axis=1,keepdims=True)
    names=["root","left","right","cloth"]
    class Source:
        def data(self,name):return {"W":weights.copy(),"joint_names":names}
    def no_seam_calculation(*args,**kwargs):
        raise AssertionError("Unchanged authored skinning needs no seam arithmetic")
    monkeypatch.setattr(p,"_source_proven_cross_mesh_seam_pairs",no_seam_calculation)
    skin={"cloth":{"weights":weights.copy(),"joint_names":names}}
    result,report=p._preserve_source_shared_seam_skinning(Source(),{},skin)
    np.testing.assert_array_equal(result["cloth"]["weights"],weights)
    assert report["changed_vertices"]==0


def test_unresolved_fit_cannot_be_published_as_success():
    with pytest.raises(ValueError,match="still intersects.*3 triangles"):
        p._validate_final_fit_clearance({"final_target_body_clearance":{"unresolved_penetrating_face_count":3,"minimum_contact_sample_after_mm":-.8}})
    p._validate_final_fit_clearance({"final_target_body_clearance":{"unresolved_penetrating_face_count":0}})


def test_final_cutaways_require_source_evidence_and_fitted_coverage():
    vertices=np.asarray([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[.02,0.,0.],[.02,.01,0.]])
    faces=np.asarray([[0,1,2],[1,3,4]])
    class Source:
        def data(self,name):return {"F":faces[:1]}
    pair={"slot":"Chest","target_literal_V":vertices,"target_literal_F":faces,
          "target_mesh_records":[{"mesh_index":2,"face_offset":0,"face_count":2}]}
    plan={"native_suppression":[{"slot":"Chest","native_mesh":2,"triangles":[0,1]}],"slots":[{"slot":"Chest"}]}
    cache={"slot_pairs":[pair]}
    result=p._refine_source_body_suppression_after_fit(Source(),{"cloth":vertices+[0.,0.,.001]},cache,plan)
    assert result["native_suppression"]==[{"slot":"Chest","native_mesh":2,"triangles":[0]}]
    assert plan["native_suppression"][0]["triangles"]==[0,1]
    assert result["fitted_coverage"]["retained_at_openings"]==1
    # A fitted garment alone must never invent body removal.
    complete={"native_suppression":[]}
    assert p._refine_source_body_suppression_after_fit(Source(),{"cloth":vertices},cache,complete)==complete
