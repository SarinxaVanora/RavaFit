from __future__ import annotations

import numpy as np

from production_b14 import _source_relative_body_penetration_state, _veto_worsened_source_relative_body_penetration


class _Source:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self._data={"mesh": {"V": np.asarray(vertices,float), "F": np.asarray(faces,np.int64)}}
    def data(self, name: str):
        return self._data[name]


def _body_triangles() -> np.ndarray:
    vertices=np.array([[-1.0,-1.0,0.0],[1.0,-1.0,0.0],[1.0,1.0,0.0],[-1.0,1.0,0.0]],float)
    faces=np.array([[0,1,2],[0,2,3]],np.int64)
    return vertices[faces]


def test_structural_body_veto_backtracks_only_post_fit_delta_that_crosses_body():
    source_vertices=np.array([[-.20,0.0,.010],[.20,0.0,.010],[.20,.20,.010],[-.20,.20,.010]],float)
    faces=np.array([[0,1,2],[0,2,3]],np.int64);source=_Source(source_vertices,faces);body=_body_triangles()
    before=source_vertices.copy();proposed=before.copy();proposed[0,2]=-.006;proposed[1,2]=-.004
    out,report=_veto_worsened_source_relative_body_penetration(source,{"mesh":before},{"mesh":proposed},body,body)
    before_state=_source_relative_body_penetration_state(source_vertices,before,body,body)
    final_state=_source_relative_body_penetration_state(source_vertices,out["mesh"],body,body)
    assert before_state["count"]==0
    assert final_state["count"]==0
    assert report["materially_worsened_before_veto"]==2
    assert report["materially_worsened_after_veto"]==0
    assert report["restricted_vertex_count"]>=2
    assert np.linalg.norm(out["mesh"]-before,axis=1).max()>0.0  # retains some safe structural authority


def test_structural_body_veto_is_exact_noop_when_proposal_does_not_worsen_body_relation():
    source_vertices=np.array([[-.20,0.0,.010],[.20,0.0,.010],[.20,.20,.010],[-.20,.20,.010]],float)
    faces=np.array([[0,1,2],[0,2,3]],np.int64);source=_Source(source_vertices,faces);body=_body_triangles()
    before=source_vertices.copy();proposed=before.copy();proposed[:,0]+=np.array([-.01,.01,.01,-.01])
    out,report=_veto_worsened_source_relative_body_penetration(source,{"mesh":before},{"mesh":proposed},body,body)
    assert np.array_equal(out["mesh"],proposed)
    assert report["restricted_vertex_count"]==0
    assert report["materially_worsened_before_veto"]==0
    assert report["materially_worsened_after_veto"]==0
