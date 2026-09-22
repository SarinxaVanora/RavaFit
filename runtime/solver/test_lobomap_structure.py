import sys
import unittest
from pathlib import Path

import numpy as np

from lobomap_production import LoBoMapMeshInput, initialise_ffxiv_source, initialise_from_body_correspondence
from lobomap_refinement import measure_structural_diagnostics, refine_ffxiv_batch, refine_ffxiv_batch_structured
from lobomap_structure import StructuralSolveConfig, TargetCollisionSurface, apply_authored_structure, apply_target_collision, collect_target_collision_surface, infer_garment_structure


class LoBoMapAuthoredStructureTests(unittest.TestCase):
    def _single_bone_cache(self):
        X=np.asarray([[-1.,-1.,0.],[1.,-1.,0.],[-1.,1.,0.],[1.,1.,0.],[-1.,-1.,1.],[1.,-1.,1.],[-1.,1.,1.],[1.,1.,1.]],dtype=float)
        Y=X+np.asarray([.20,.05,.10]);BW=np.ones((len(X),1));N=np.tile([0.,0.,1.],(len(X),1))
        return {"X":X,"Y":Y,"BW":BW,"NS":N,"NT":N,"names":["root"]}

    def test_tiny_connected_component_is_rigid_and_projection_preserves_uniform_authored_shape(self):
        cache=self._single_bone_cache();names=["root"]
        V=np.asarray([[0.,0.,.01],[.01,0.,.01],[0.,.01,.01],[.01,.01,.01]])
        F=np.asarray([[0,1,2],[1,3,2]],dtype=np.int64);W=np.ones((4,1))
        initial,mapping,_,_=initialise_from_body_correspondence(V,W,names,cache["X"],cache["Y"],cache["BW"],names)
        mesh=LoBoMapMeshInput("detail",V,F,W,tuple(names),np.arange(4),4,"m")
        structure,contexts,_=infer_garment_structure({"detail":mesh},{"detail":mapping},cache)
        self.assertEqual(len(structure.components),1);self.assertEqual(structure.components[0].structural_class,"rigid")
        candidate=initial.copy();candidate[3]+=np.asarray([.006,-.003,.004])
        out,_=apply_authored_structure({"detail":mesh},{"detail":initial},{"detail":candidate},structure,contexts)
        edges=np.asarray([[0,1],[0,2],[1,3],[2,3],[1,2]],dtype=np.int64)
        src=np.linalg.norm(V[edges[:,0]]-V[edges[:,1]],axis=1);solved=np.linalg.norm(out["detail"][edges[:,0]]-out["detail"][edges[:,1]],axis=1)
        ratio=solved/src
        self.assertLess(float(np.max(ratio)-np.min(ratio)),1e-9)

    def test_mirrored_disconnected_peers_share_stable_layer(self):
        names=["root"]
        # Bilateral in X, deliberately not front/back or vertical symmetric so the
        # geometry-derived mirror plane is unambiguous.
        half=np.asarray([[.45,-.80,.02],[.52,-.15,.10],[.38,.45,.18],[.22,.95,.06],[.48,.20,.32],[.30,.70,.40]])
        X=np.vstack((half,half*np.asarray([-1.,1.,1.])));Y=X+np.asarray([.20,.05,.10]);BW=np.ones((len(X),1));N=np.tile([0.,0.,1.],(len(X),1))
        cache={"X":X,"Y":Y,"BW":BW,"NS":N,"NT":N,"names":names}
        left=np.asarray([[-.30,0.,.03],[-.28,0.,.03],[-.30,.02,.03]])
        right=left.copy();right[:,0]*=-1.0
        V=np.vstack((left,right));F=np.asarray([[0,1,2],[3,4,5]],dtype=np.int64);W=np.ones((6,1))
        initial,mapping,_,_=initialise_from_body_correspondence(V,W,names,cache["X"],cache["Y"],cache["BW"],names)
        mesh=LoBoMapMeshInput("peers",V,F,W,tuple(names),np.arange(6),6,"shared")
        structure,_,report=infer_garment_structure({"peers":mesh},{"peers":mapping},cache)
        self.assertEqual(report["component_count"],2);self.assertEqual(report["layer_count"],1)
        self.assertEqual(set(structure.layers[0].component_ids),set(x.stable_id for x in structure.components))

    def test_literal_triangle_collision_clears_face_interior_not_only_vertices(self):
        V=np.asarray([[-.01,-.01,-.001],[.01,-.01,-.001],[0.,.01,-.001]])
        F=np.asarray([[0,1,2]],dtype=np.int64);W=np.ones((3,1));mesh=LoBoMapMeshInput("piece",V,F,W,("root",),np.arange(3),3,"m")
        # Source support plane sits 1 mm below the garment, so collision is eligible.
        X=np.asarray([[-1.,-1.,-.002],[1.,-1.,-.002],[-1.,1.,-.002],[1.,1.,-.002],[0.,-.5,-.002],[0.,.5,-.002]])
        Y=X.copy();Y[:,2]=0.0
        cache={"X":X,"Y":Y,"BW":np.ones((len(X),1)),"NS":np.tile([0.,0.,1.],(len(X),1)),"NT":np.tile([0.,0.,1.],(len(X),1)),"names":["root"]}
        initial,mapping,_,_=initialise_from_body_correspondence(V,W,["root"],cache["X"],cache["Y"],cache["BW"],["root"])
        structure,contexts,_=infer_garment_structure({"piece":mesh},{"piece":mapping},cache)
        surface=TargetCollisionSurface.build(np.asarray([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]]),np.asarray([[0,1,2]],dtype=np.int64),np.tile([0.,0.,1.],(3,1)))
        penetrating={"piece":V.copy()}
        out,report=apply_target_collision({"piece":mesh},penetrating,structure,contexts,surface,initial_positions={"piece":initial})
        self.assertEqual(report["remaining_penetrations"],0)
        self.assertTrue(np.all(out["piece"][:,2]>=-1e-7))


class LoBoMapRealFixtureStructureTests(unittest.TestCase):
    def test_real_fixture_structure_removes_flips_collapse_and_literal_penetration(self):
        root=Path(__file__).resolve().parents[1];scripts=root/"b14_frozen"/"scripts"
        fixture=root/"b14_frozen"/"inputs"/"Original_outfit.glb";cache_path=root/"b14_frozen"/"workers"/"body_cache.npz"
        if not fixture.exists() or not cache_path.exists():self.skipTest("embedded B14 fixture is unavailable")
        if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
        from ffxiv_lobofit import GLB
        source=GLB(fixture);packed=np.load(cache_path,allow_pickle=True);cache={key:packed[key] for key in ("X","Y","BW","NS","NT","names")}
        body={"mesh 0.1","mesh 0.2","mesh 0.3"}
        initial,mappings,views,_=initialise_ffxiv_source(source,cache,body_mesh_names=body)
        baseline,baseline_report=refine_ffxiv_batch(views,mappings,initial,cache)
        surface,_=collect_target_collision_surface(source,cache,body)
        final,report=refine_ffxiv_batch_structured(views,mappings,initial,cache,collision_surface=surface)
        self.assertEqual(report["remaining_actual_penetrations"],0)
        self.assertGreater(report["structure_inference"]["component_count"],100)
        self.assertGreaterEqual(report["structure_inference"]["layer_count"],2)
        self.assertLess(report["mesh_p95_clearance_error_final_max_mm"],2.0)
        self.assertEqual(report["initial_source_topology"]["violations_after"],0)
        self.assertEqual(report["post_structure_source_topology"]["violations_after"],0)
        for name in views:
            structural=report["meshes"][name]["structural"]
            self.assertEqual(structural["source_triangle_flip_count"],0,name)
            self.assertEqual(structural["source_area_floor_violation_count_ratio_lt_0_01"],0,name)
            self.assertEqual(structural["near_degenerate_triangle_count_ratio_lt_0_10"],0,name)
            self.assertTrue(np.all(np.isfinite(final[name])))
        base_top=baseline_report["meshes"]["mesh 1.0"]["structural"]["initial_to_refined_edge_strain_p95_pct"]
        final_top=report["meshes"]["mesh 1.0"]["structural"]["initial_to_refined_edge_strain_p95_pct"]
        self.assertLess(final_top,base_top*.60)
        base_bra=baseline_report["meshes"]["mesh 4.0"]["structural"]["initial_to_refined_edge_strain_p95_pct"]
        final_bra=report["meshes"]["mesh 4.0"]["structural"]["initial_to_refined_edge_strain_p95_pct"]
        self.assertLess(final_bra,base_bra*.80)


if __name__ == "__main__":unittest.main()
