from __future__ import annotations
import numpy as np
import production_b14 as prod


def test_support_equivalent_source_preserve_accepts_near_identical_supports():
    cache={
        'slot_stats':[
            {'slot':'Feet','displacement_rms_mm':2.99,'raw_displacement_rms_mm':2.99,'chosen_spatial_p95_mm':5.21},
        ],
        'dense_cross_sex_source_proxy':False,
    }
    report=prod._support_equivalent_source_preserve_decision(cache)
    assert report['eligible'] is True


def test_support_equivalent_source_preserve_requires_every_region_to_be_stable():
    cache={
        'slot_stats':[
            {'slot':'Hands','displacement_rms_mm':1.4,'raw_displacement_rms_mm':1.4,'chosen_spatial_p95_mm':2.6},
            {'slot':'Chest','displacement_rms_mm':8.0,'raw_displacement_rms_mm':8.0,'chosen_spatial_p95_mm':12.0},
        ],
        'dense_cross_sex_source_proxy':False,
    }
    report=prod._support_equivalent_source_preserve_decision(cache)
    assert report['eligible'] is False


def test_support_equivalent_source_preserve_rejects_cross_sex_dense_bridge():
    cache={
        'slot_stats':[
            {'slot':'Legs','displacement_rms_mm':1.0,'raw_displacement_rms_mm':1.0,'chosen_spatial_p95_mm':2.0},
        ],
        'dense_cross_sex_source_proxy':True,
    }
    report=prod._support_equivalent_source_preserve_decision(cache)
    assert report['eligible'] is False


class _FakeSource:
    def mesh_names(self):
        return ['mesh 0','mesh 1']
    def data(self,name):
        if name=='mesh 0':
            return {'V':np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),'F':np.array([[0,1,2]],dtype=np.int64),'W':np.ones((3,1)),'joint_names':['j_root']}
        return {'V':np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]),'F':np.array([[0,1,2]],dtype=np.int64),'W':np.ones((3,1)),'joint_names':['j_root']}


def test_source_preserved_solution_excludes_body_and_keeps_geometry_and_weights_exact():
    src=_FakeSource()
    positions,skinning,records,stats=prod._source_preserved_garment_solution(src,{'mesh 0'})
    assert list(positions)==['mesh 1']
    assert np.array_equal(positions['mesh 1'],src.data('mesh 1')['V'])
    assert np.array_equal(skinning['mesh 1']['weights'],src.data('mesh 1')['W'])
    assert records['mesh 1']['mode']=='source_preserved_support_equivalent'
    assert stats['post_b14_geometry_mutation'] is False
