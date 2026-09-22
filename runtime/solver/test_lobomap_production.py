import unittest
from pathlib import Path
import sys
import numpy as np

from lobomap_production import build_bone_frames, collect_ffxiv_mesh_batch, compile_lobomap, decode_lobomap, freeze_ffxiv_solution, initialise_ffxiv_source, initialise_garment, remap_weights, restore_rendered_positions


class LoBoMapProductionTests(unittest.TestCase):
    def _frames(self, scale=1.0, hip_shift=(0.0,0.0,0.0)):
        names=("root","thigh_l","shin_l","thigh_r","shin_r")
        P=np.asarray([
            [0.0,0.0,0.0],[-0.16,-0.05,0.0],[-0.17,-0.55,0.0],[0.16,-0.05,0.0],[0.17,-0.55,0.0]
        ],dtype=float)*scale
        P+=np.asarray(hip_shift,float)
        parents=np.asarray([-1,0,1,0,3],dtype=np.int64)
        return build_bone_frames(names,P,parents,root_reference=np.asarray([1.0,0.0,0.0]))

    def test_source_roundtrip_is_exact_for_single_bone_vertices(self):
        f=self._frames();V=np.asarray([[-.16,-.20,.05],[-.17,-.42,-.03],[.16,-.20,.05],[.17,-.42,-.03]],float)
        W=np.zeros((4,5),float);W[0:2,1]=1.0;W[2:4,3]=1.0
        m=compile_lobomap(V,W,f.bone_names,f,max_influences=4)
        self.assertTrue(np.allclose(decode_lobomap(m,f),V,atol=1e-11,rtol=0.0))

    def test_whole_garment_moves_coherently_when_target_bones_change(self):
        src=self._frames();tgt=self._frames(scale=1.2,hip_shift=(0.0,.10,.0))
        V=np.asarray([[-.16,-.20,.05],[-.17,-.42,-.03],[.16,-.20,.05],[.17,-.42,-.03]],float)
        W=np.zeros((4,5),float);W[:2,1]=1.0;W[2:,3]=1.0
        out,_,report=initialise_garment(V,W,src.bone_names,src,tgt)
        self.assertEqual(out.shape,V.shape);self.assertGreater(report.displacement_p95_mm,1.0)
        # Bilateral authored identity must survive coherent mapping.
        self.assertTrue(np.all(out[:2,0]<0.0));self.assertTrue(np.all(out[2:,0]>0.0))

    def test_blended_vertex_uses_multiple_bone_local_frames(self):
        src=self._frames();tgt=self._frames();
        # Move only the left shin joint in target skeleton.
        P=tgt.origins.copy();P[2]+=[-.08,-.10,.02];tgt=build_bone_frames(tgt.bone_names,P,tgt.parents,root_reference=np.asarray([1.,0.,0.]))
        V=np.asarray([[-.165,-.35,.02]],float);W=np.zeros((1,5));W[0,1]=.5;W[0,2]=.5
        mapping=compile_lobomap(V,W,src.bone_names,src,max_influences=4);out=decode_lobomap(mapping,tgt)
        self.assertTrue(np.all(np.isfinite(out)));self.assertGreater(float(np.linalg.norm(out-V)),1e-4)
        self.assertEqual(int(np.count_nonzero(mapping.influence_weights[0]>0.0)),2)

    def test_joint_name_remap_is_authoritative(self):
        W=np.asarray([[.25,.75]],float)
        out=remap_weights(W,["b","a"],["a","b"])
        self.assertTrue(np.allclose(out,[[.75,.25]]))

    def test_missing_all_influences_fails_instead_of_guessing(self):
        with self.assertRaises(ValueError):remap_weights(np.asarray([[1.0]]),["missing"],["root"])

    def test_frame_axes_are_orthonormal_and_hierarchically_stable(self):
        f=self._frames();gram=np.einsum("bij,bik->bjk",f.axes,f.axes)
        self.assertTrue(np.allclose(gram,np.eye(3)[None,:,:],atol=2e-6,rtol=0.0))
        self.assertTrue(np.all(f.lengths>0.0))


class LoBoMapBodyCorrespondenceTests(unittest.TestCase):
    def test_body_derived_frames_recover_bilateral_similarity_motion(self):
        from lobomap_production import build_body_correspondence_frames, initialise_from_body_correspondence
        names=("left","right")
        # Two clearly separated body regions with distinct transforms.
        left=np.asarray([[-1.,0.,0.],[-1.,1.,0.],[-1.,0.,1.],[-1.,1.,1.]])
        right=np.asarray([[1.,0.,0.],[1.,1.,0.],[1.,0.,1.],[1.,1.,1.]])
        X=np.vstack([left,right]);BW=np.zeros((8,2));BW[:4,0]=1.;BW[4:,1]=1.
        Y=X.copy();Y[:4]=np.asarray([-.2,.1,.0])+1.2*(X[:4]-np.asarray([-1.,0.,0.]));Y[4:]=np.asarray([1.4,-.1,.0])+.8*(X[4:]-np.asarray([1.,0.,0.]))
        sf,tf,rep=build_body_correspondence_frames(X,Y,BW,names)
        self.assertEqual(int(rep["supported_bone_count"]),2)
        garment=np.asarray([[-1.,.5,.5],[1.,.5,.5]])
        GW=np.eye(2)
        out,_,_,_=initialise_from_body_correspondence(garment,GW,names,X,Y,BW,names,max_influences=2)
        expected=np.asarray([[-.2,.7,.6],[1.4,.3,.4]])
        self.assertTrue(np.allclose(out,expected,atol=1e-10,rtol=0.0))

    def test_body_correspondence_lane_is_component_agnostic(self):
        from lobomap_production import initialise_from_body_correspondence
        names=("leg",)
        X=np.asarray([[0.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,1.,1.]])
        Y=np.asarray([[.1,0.,0.],[.1,1.1,0.],[.1,0.,1.1],[.1,1.1,1.1]])
        BW=np.ones((4,1));V=np.asarray([[0.,.2,.2],[0.,.8,.8],[0.,.2,.8],[0.,.8,.2]])
        # Deliberately four disconnected 'details': the initialiser must not need topology.
        out,_,_,_=initialise_from_body_correspondence(V,np.ones((4,1)),names,X,Y,BW,names)
        self.assertTrue(np.allclose(out[:,0],.1,atol=1e-10));self.assertTrue(np.all(out[:,1:]>=0.0))

class LoBoMapBatchTests(unittest.TestCase):
    def test_disconnected_meshes_share_one_body_frame_field(self):
        from lobomap_production import initialise_mesh_batch
        names=("leg",);X=np.asarray([[0.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,1.,1.]])
        Y=X+np.asarray([.25,.10,-.05]);BW=np.ones((4,1))
        cache={"X":X,"Y":Y,"BW":BW,"names":list(names)}
        meshes={
            "band":{"V":np.asarray([[0.,.2,.2],[0.,.3,.2]]),"W":np.ones((2,1)),"joint_names":list(names)},
            "coin":{"V":np.asarray([[0.,.8,.8],[0.,.81,.8]]),"W":np.ones((2,1)),"joint_names":list(names)},
        }
        out,_,report=initialise_mesh_batch(meshes,cache)
        delta=np.asarray([.25,.10,-.05])
        self.assertTrue(np.allclose(out["band"]-meshes["band"]["V"],delta,atol=1e-10))
        self.assertTrue(np.allclose(out["coin"]-meshes["coin"]["V"],delta,atol=1e-10))
        self.assertEqual(report["mesh_count"],2)


class LoBoMapFfxivAdapterTests(unittest.TestCase):
    class _FakeSource:
        def __init__(self, rows):self.rows=rows
        def mesh_names(self):return list(self.rows)
        def data(self,name):
            value=self.rows[name]
            if isinstance(value,Exception):raise value
            return value

    def test_raw_position_restore_preserves_dead_storage_rows(self):
        raw=np.asarray([[0.,0.,0.],[9.,9.,9.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float32);ids=np.asarray([0,2,3],dtype=np.int64);solved=np.asarray([[.1,0.,0.],[1.1,0.,0.],[.1,1.,0.]],dtype=np.float64)
        out=restore_rendered_positions(raw,ids,solved)
        self.assertTrue(np.array_equal(out[1],raw[1]));self.assertTrue(np.array_equal(out[ids],solved.astype(np.float32)))

    def test_real_fixture_freeze_roundtrips_positions_and_preserves_skinning(self):
        import tempfile
        root=Path(__file__).resolve().parents[1];scripts=root/"b14_frozen"/"scripts";fixture=root/"b14_frozen"/"inputs"/"Original_outfit.glb";cache_path=root/"b14_frozen"/"workers"/"body_cache.npz"
        if not fixture.exists() or not cache_path.exists():self.skipTest("embedded B14 fixture is unavailable")
        if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
        from ffxiv_lobofit import GLB
        source=GLB(fixture);packed=np.load(cache_path,allow_pickle=True);cache={key:packed[key] for key in ("X","Y","BW","names")};body={"mesh 0.1","mesh 0.2","mesh 0.3"}
        _,_,views,_=initialise_ffxiv_source(source,cache,body_mesh_names=body);solved={name:view.vertices+np.asarray([1e-5,0.,0.]) for name,view in views.items()}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"frozen.glb";report=freeze_ffxiv_solution(fixture,out,views,solved)
            self.assertTrue(out.exists());self.assertEqual(report["mesh_count"],8);self.assertGreater(report["skinning_accessors_verified"],0)

    def test_adapter_excludes_body_helpers_and_dead_storage_vertices(self):
        names=["root"]
        garment={
            "V":np.asarray([[0.,0.,0.],[9.,9.,9.],[1.,0.,0.],[0.,1.,0.]],float),
            "F":np.asarray([[0,2,3]],np.int64),
            "W":np.ones((4,1),float),
            "joint_names":names,
            "material":"garment",
        }
        body={"V":np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),"F":np.asarray([[0,1,2]]),"W":np.ones((3,1)),"joint_names":names}
        source=self._FakeSource({"body":body,"garment":garment,"helper":RuntimeError("no skin")})
        meshes,report=collect_ffxiv_mesh_batch(source,body_mesh_names={"body"})
        self.assertEqual(set(meshes),{"garment"})
        view=meshes["garment"]
        self.assertEqual(view.raw_vertex_count,4);self.assertEqual(view.vertex_count,3)
        self.assertTrue(np.array_equal(view.raw_vertex_ids,np.asarray([0,2,3])))
        self.assertTrue(np.array_equal(view.faces,np.asarray([[0,1,2]])))
        self.assertEqual(report["meshes"]["garment"]["unreferenced_storage_vertices"],1)
        self.assertEqual(report["meshes"]["body"]["reason"],"source body mesh")
        self.assertIn("not skinned garment geometry",report["meshes"]["helper"]["reason"])

    def test_mesh_filter_is_applied_before_payload_adaptation(self):
        row={"V":np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),"F":np.asarray([[0,1,2]]),"W":np.ones((3,1)),"joint_names":["root"]}
        source=self._FakeSource({"a":row,"b":RuntimeError("should not be read")})
        meshes,report=collect_ffxiv_mesh_batch(source,mesh_filter={"a"})
        self.assertEqual(set(meshes),{"a"});self.assertEqual(report["meshes"]["b"]["reason"],"excluded by mesh filter")

    def test_real_embedded_b14_fixture_uses_exact_glb_payload_contract(self):
        root=Path(__file__).resolve().parents[1]
        scripts=root/"b14_frozen"/"scripts"
        fixture=root/"b14_frozen"/"inputs"/"Original_outfit.glb"
        cache_path=root/"b14_frozen"/"workers"/"body_cache.npz"
        if not fixture.exists() or not cache_path.exists():self.skipTest("embedded B14 fixture is unavailable")
        if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
        from ffxiv_lobofit import GLB
        source=GLB(fixture);packed=np.load(cache_path,allow_pickle=True)
        cache={key:packed[key] for key in ("X","Y","BW","names")}
        # Fixture-specific body ids are supplied as data, never encoded in production logic.
        body={"mesh 0.1","mesh 0.2","mesh 0.3"}
        positions,mappings,views,report=initialise_ffxiv_source(source,cache,body_mesh_names=body)
        self.assertEqual(len(positions),8);self.assertEqual(len(mappings),8);self.assertEqual(len(views),8)
        self.assertEqual(report["vertex_count"],38471)
        self.assertEqual(report["ffxiv_adapter"]["rendered_vertex_count"],38471)
        self.assertTrue(all(value["reconstruction_error_max_mm"]<1e-8 for value in report["meshes"].values()))


if __name__ == "__main__":unittest.main()
