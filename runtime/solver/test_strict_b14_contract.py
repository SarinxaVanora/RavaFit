import numpy as np
import production_b14 as p


def _ref(V, UV, N, W, names):
    return {"V":np.asarray(V,float),"UV":np.asarray(UV,float),"N":np.asarray(N,float),"W":np.asarray(W,float),"joint_names":list(names),"slot":"top"}


def test_strict_uv_contract_reproduces_target_uv_mapping():
    src_v=np.asarray([[0,0,0],[1,0,0],[0,1,0]],float)
    tgt_v=src_v+np.asarray([0.25,-0.5,0.75])
    uv=np.asarray([[0,0],[1,0],[0,1]],float)
    n=np.tile(np.asarray([[0,0,1.]],float),(3,1))
    w=np.asarray([[1,0],[0.5,0.5],[0,1]],float)
    out=p._apply_strict_historical_uv_contract([(_ref(src_v,uv,n,w,["a","b"]),_ref(tgt_v,uv,n,w,["a","b"]))],{})
    assert out["_ravafit_strict_b14_contract"] is True
    np.testing.assert_allclose(out["X"],src_v,atol=0,rtol=0)
    np.testing.assert_allclose(out["Y"],tgt_v,atol=1e-6,rtol=0)
    np.testing.assert_allclose(out["BW"],w,atol=1e-12,rtol=0)


def test_indexed_render_view_excludes_dead_storage():
    data={"V":np.asarray([[0,0,0],[1,0,0],[0,1,0],[50,50,50],[60,60,60]],float),"F":np.asarray([[0,1,2]],np.int64),"W":np.ones((5,1)),"N":np.ones((5,3)),"UV":np.ones((5,2)),"joint_names":["a"],"material":"x"}
    indexed,report=p._indexed_render_view(data)
    assert report["raw_vertices"]==5
    assert report["indexed_vertices"]==3
    assert report["unindexed_vertices"]==2
    assert indexed["V"].shape==(3,3)
    np.testing.assert_array_equal(indexed["F"],np.asarray([[0,1,2]]))


def test_indexed_solution_expands_without_touching_dead_rows():
    raw={"V":np.asarray([[0,0,0],[1,0,0],[0,1,0],[50,50,50]],float),"F":np.asarray([[0,1,2]],np.int64),"W":np.asarray([[1.],[1.],[1.],[7.]]),"N":np.ones((4,3)),"UV":np.ones((4,2)),"joint_names":["a"],"material":"x"}
    class S:
        js={}
        def mesh_names(self):return ["m"]
        def data(self,name):return raw
    view=p._IndexedRenderGarmentSource(S(),set())
    solved=view.data("m")["V"]+1
    pos,skin=p._collapse_indexed_render_garment_solution(view,{"m":solved},{"m":{"weights":np.asarray([[2.],[2.],[2.]]),"joint_names":["a"],"stage":{}}})
    np.testing.assert_allclose(pos["m"][:3],raw["V"][:3]+1)
    np.testing.assert_allclose(pos["m"][3],raw["V"][3])
    np.testing.assert_allclose(skin["m"]["weights"][:3],2)
    np.testing.assert_allclose(skin["m"]["weights"][3],7)


def test_strict_macro_body_surface_prefers_broad_material_and_compacts_dead_vertices():
    ref={
        "V":np.asarray([[0,0,0],[2,0,0],[0,3,0],[99,99,99],[0,0,0],[.1,0,0],[0,.1,0]],float),
        "F":np.asarray([[0,1,2],[4,5,6]],np.int64),
        "UV":np.zeros((7,2),float),
        "N":np.tile(np.asarray([[0,0,1.]],float),(7,1)),
        "W":np.ones((7,1),float),
        "joint_names":["a"],
        "slot":"legs",
        "mesh_records":[
            {"mesh_index":0,"material":"/macro.mtrl","vertex_offset":0,"vertex_count":4,"face_offset":0,"face_count":1,"index_count":3},
            {"mesh_index":1,"material":"/detail.mtrl","vertex_offset":4,"vertex_count":3,"face_offset":1,"face_count":1,"index_count":3},
        ],
    }
    out,report=p._strict_macro_body_surface(ref)
    assert report["enabled"] is True
    assert report["selected_material"]=="/macro.mtrl"
    assert report["selected_vertices"]==3
    np.testing.assert_allclose(out["V"],ref["V"][:3])
    np.testing.assert_array_equal(out["F"],np.asarray([[0,1,2]],np.int64))


def test_strict_uv_map_repairs_only_gross_rig_ambiguous_duplicate_uv():
    SV=np.asarray([[0,0,0]],float)
    SUV=np.asarray([[0,0]],float)
    SN=np.asarray([[0,0,1]],float)
    SW=np.asarray([[1,0]],float)
    TV=np.asarray([[.01,0,0],[1.0,0,0]],float)
    TUV=np.asarray([[0,0],[0,0]],float)
    TN=np.asarray([[0,0,1],[0,0,1]],float)
    TW=np.asarray([[1,0],[0,1]],float)
    Y,NT,report=p._strict_uv_map(SV,SUV,SN,SW,TV,TUV,TN,TW)
    assert report["rig_ambiguity_repaired_vertices"]==1
    assert float(Y[0,0])<0.05
    np.testing.assert_allclose(NT,np.asarray([[0,0,1]],float),atol=1e-12)


def test_strict_b14_surface_sanitisation_drops_only_unusable_faces():
    tri=np.asarray([
        [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
        [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],
        [[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]],
        [[np.nan,0.,0.],[0.,1.,0.],[0.,0.,1.]],
    ],dtype=np.float64)
    cleaned,report=p._sanitise_strict_b14_surface_triangles(tri,"target-test")
    assert cleaned.shape==(1,3,3)
    np.testing.assert_array_equal(cleaned[0],tri[0])
    assert report["input_triangle_count"]==4
    assert report["output_triangle_count"]==1
    assert report["dropped_non_finite_triangle_count"]==1
    assert report["dropped_degenerate_triangle_count"]==2
    assert report["changed"] is True
