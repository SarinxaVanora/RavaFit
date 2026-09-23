import sys
import unittest
from pathlib import Path

import numpy as np

from lobomap_production import LoBoMapMeshInput, initialise_ffxiv_source, initialise_from_body_correspondence
from lobomap_refinement import LoBoMapRefinementConfig, measure_structural_diagnostics, refine_ffxiv_batch, refine_mesh_local_residuals


class LoBoMapLocalRefinementTests(unittest.TestCase):
    def _plane_case(self, source_clearance=.010):
        names=["root"]
        X=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        Y=X+np.asarray([.2,.1,.0]);BW=np.ones((4,1));NS=np.tile([0.,0.,1.],(4,1));NT=NS.copy()
        V=np.asarray([[.2,.2,source_clearance],[.8,.2,source_clearance],[.2,.8,source_clearance],[.8,.8,source_clearance]])
        F=np.asarray([[0,1,2],[1,3,2]],dtype=np.int64);W=np.ones((4,1))
        initial,mapping,_,_=initialise_from_body_correspondence(V,W,names,X,Y,BW,names)
        mesh=LoBoMapMeshInput("garment",V,F,W,tuple(names),np.arange(4),4)
        cache={"X":X,"Y":Y,"BW":BW,"NS":NS,"NT":NT,"names":names}
        return mesh,mapping,initial,cache

    def test_refinement_recovers_source_signed_clearance_from_perturbed_initialisation(self):
        mesh,mapping,initial,cache=self._plane_case(.010)
        perturbed=initial.copy();perturbed[:,2]-=.020
        out,report,_,_=refine_mesh_local_residuals(mesh,mapping,perturbed,cache)
        self.assertTrue(np.allclose(out[:,2],.010,atol=1e-10))
        self.assertGreater(report["clearance_error_before_p95_mm"],19.9)
        self.assertLess(report["clearance_error_after_p95_mm"],1e-7)
        # The correction is normal-only in this planar case, so authored in-plane shape survives exactly.
        self.assertTrue(np.allclose(out[:,:2],perturbed[:,:2],atol=1e-12))

    def test_negative_authored_clearance_is_not_rewritten_by_contact_guard(self):
        mesh,mapping,initial,cache=self._plane_case(-.005)
        out,report,_,_=refine_mesh_local_residuals(mesh,mapping,initial,cache)
        self.assertTrue(np.allclose(out,initial,atol=1e-12))
        self.assertEqual(report["contact_guard_vertices"],0)

    def test_residual_cap_limits_extreme_local_correction(self):
        mesh,mapping,initial,cache=self._plane_case(.010)
        perturbed=initial.copy();perturbed[:,2]-=.200
        cfg=LoBoMapRefinementConfig(max_residual_m=.030,coarse_iterations=0,fine_iterations=0)
        out,report,_,_=refine_mesh_local_residuals(mesh,mapping,perturbed,cache,config=cfg)
        # The final zero-penetration guard may add enough to reach the support surface, but not the full
        # authored 10 mm clearance after a deliberately extreme perturbation.
        self.assertTrue(np.all(out[:,2]>=-1e-12))
        self.assertTrue(np.all(out[:,2]<.010-1e-5))
        self.assertGreater(report["clearance_error_after_p95_mm"],0.0)


    def test_structural_diagnostics_are_identity_for_unchanged_geometry(self):
        mesh,_,initial,_=self._plane_case(.010)
        report=measure_structural_diagnostics(mesh,mesh.vertices,mesh.vertices)
        self.assertAlmostEqual(report["source_to_refined_edge_strain_p95_pct"],0.0,places=10)
        self.assertAlmostEqual(report["initial_area_ratio_min"],1.0,places=10)
        self.assertAlmostEqual(report["laplacian_delta_p95_mm"],0.0,places=10)
        self.assertEqual(report["near_degenerate_triangle_count_ratio_lt_0_10"],0)


class LoBoMapRealFixtureRefinementTests(unittest.TestCase):
    def test_real_b14_fixture_local_refinement_reduces_body_relative_fit_error(self):
        root=Path(__file__).resolve().parents[1];scripts=root/"b14_frozen"/"scripts"
        fixture=root/"b14_frozen"/"inputs"/"Original_outfit.glb";cache_path=root/"b14_frozen"/"workers"/"body_cache.npz"
        if not fixture.exists() or not cache_path.exists():self.skipTest("embedded B14 fixture is unavailable")
        if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
        from ffxiv_lobofit import GLB
        source=GLB(fixture);packed=np.load(cache_path,allow_pickle=True)
        cache={key:packed[key] for key in ("X","Y","BW","NS","NT","names")}
        body={"mesh 0.1","mesh 0.2","mesh 0.3"}
        initial,mappings,views,_=initialise_ffxiv_source(source,cache,body_mesh_names=body)
        refined,report=refine_ffxiv_batch(views,mappings,initial,cache)
        self.assertEqual(report["vertex_count"],38471);self.assertEqual(set(refined),set(initial))
        self.assertLess(report["mesh_p95_clearance_error_after_max_mm"],report["mesh_p95_clearance_error_before_max_mm"])
        self.assertLess(report["mesh_p95_clearance_error_after_max_mm"],.20)
        self.assertTrue(all(np.all(np.isfinite(value)) for value in refined.values()))


if __name__ == "__main__":unittest.main()
